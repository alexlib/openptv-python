
import shutil
from pathlib import Path
import tempfile
import pytest
import filecmp

from openptv_python.parameters import (
    read_sequence_par, read_control_par, read_track_par, convert_track_par_to_tuple
)
from openptv_python.calibration import read_ori
from openptv_python.orientation import point_position
from openptv_python.track import track_forward_start, trackcorr_c_loop, trackcorr_c_finish  # TODO: use actual tracking functions
from openptv_python.parameters import MultimediaPar
from openptv_python.trafo import pixel_to_metric

@pytest.mark.integration
def test_sequence_and_tracking_on_burgers():
    src_dir = Path(__file__).parent.resolve() / "testing_folder" / "burgers"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        work_dir = tmp_path / "burgers"
        shutil.copytree(src_dir, work_dir)

        param_dir = work_dir / "parameters"
        cal_dir = work_dir / "cal"
        img_orig_dir = work_dir / "img_orig"
        img_dir = work_dir / "img"
        res_dir = work_dir / "res_test"

        res_dir.mkdir(exist_ok=True)

        # Copy img_orig to img for the test (ensure all *_targets files are present)
        img_dir.mkdir(exist_ok=True)
        # Copy all *_targets files from img_orig to img
        for file in img_orig_dir.glob('cam*_*_targets'):
            shutil.copy(file, img_dir / file.name)
        for frame in range(10001, 10006):
            for cam in range(1, 5):
                src = img_orig_dir / f"cam{cam}.{frame}_targets"
                dst = img_dir / f"cam{cam}.{frame}_targets"
                print(f"[DEBUG] Checking {src}: exists={src.exists()} size={src.stat().st_size if src.exists() else 'N/A'}")
                if src.exists():
                    shutil.copy(src, dst)
                else:
                    raise FileNotFoundError(f"Missing required target file: {src}")
        res_dir.mkdir(exist_ok=True)

        # Patch sequence.par and ptv.par to use img/ instead of img_orig/
        seq_par_file = param_dir / "sequence.par"
        ptv_par_file = param_dir / "ptv.par"
        with open(seq_par_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(4):
            if lines[i].startswith("img_orig/"):
                lines[i] = lines[i].replace("img_orig/", "img/")
        with open(seq_par_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        with open(ptv_par_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(1, 9, 2):
            if lines[i].startswith("img_orig/"):
                lines[i] = lines[i].replace("img_orig/", "img/")
        with open(ptv_par_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        import os
        os.chdir(work_dir)

        from openptv_python.tracking_run import TrackingRun
        from openptv_python.track import track_forward_start, trackcorr_c_loop, trackcorr_c_finish
        from openptv_python.tracking_frame_buf import read_targets, write_targets, Frame
        from openptv_python.correspondences import correspondences
        # from openptv_python.imgcoord import multitrack

        # 1. Read parameters
        seq_par = read_sequence_par(param_dir / "sequence.par", num_cams=4)
        print("[DEBUG] Loaded SequencePar:")
        print(seq_par)
        cpar = read_control_par(param_dir / "ptv.par")
        print("[DEBUG] Loaded ControlPar:")
        print(cpar)
        tpar_raw = read_track_par(param_dir / "track.par")
        print("[DEBUG] Loaded TrackPar (raw):")
        print(tpar_raw)
        tpar = convert_track_par_to_tuple(tpar_raw)
        print("[DEBUG] Loaded TrackPar (tuple):")
        print(tpar)
        from openptv_python.parameters import read_volume_par
        crit_file = param_dir / "criteria.par"
        vpar = read_volume_par(crit_file)
        print("[DEBUG] Loaded VolumePar (from criteria.par):")
        print(vpar)
        # Load calibrations for all 4 cameras
        cal = []
        for cam in range(1, 5):
            ori_file = cal_dir / f"cam{cam}.tif.ori"
            add_file = cal_dir / f"cam{cam}.tif.addpar"
            cal_obj = read_ori(ori_file, add_file)
            # Print detailed calibration parameters
            print(f"[DEBUG] Calibration cam{cam}:")
            for attr in dir(cal_obj):
                if not attr.startswith('_') and not callable(getattr(cal_obj, attr)):
                    print(f"    {attr}: {getattr(cal_obj, attr)}")
            cal.append(cal_obj)

        corres_file_base = str(res_dir / "rt_is")
        linkage_file_base = str(res_dir / "ptv_is")
        prio_file_base = str(res_dir / "added")

        # 2. Stereo-matching: generate correspondences and triangulate for each frame
        import numpy as np
        from openptv_python.tracking_frame_buf import Pathinfo, Corres_dtype
        mm = MultimediaPar()  # Use default multimedia parameters (or load if needed)
        import pprint
        from openptv_python.constants import COORD_UNUSED
        for frame in range(seq_par.first, seq_par.last + 1):
            print(f"\n[DEBUG] ===== Frame {frame} =====")
            frm = Frame(num_cams=4, max_targets=1000)
            corrected = []
            for cam in range(1, 5):
                file_base = str(img_dir / f"cam{cam}.%05d")
                targets = read_targets(file_base, frame)
                # Print the raw file content after reading
                target_file_path = img_dir / f"cam{cam}.{frame}_targets"
                try:
                    with open(target_file_path, "r", encoding="utf-8") as tf:
                        file_content = tf.read()
                    print(f"[DEBUG] cam{cam}.{frame}_targets file content:\n{file_content}")
                except Exception as e:
                    print(f"[DEBUG] Could not read cam{cam}.{frame}_targets: {e}")
                print(f"[DEBUG] cam{cam} targets for frame {frame}: {pprint.pformat([{'x': t.x, 'y': t.y, 'pnr': t.pnr} for t in targets])}")
                # Sort targets by X coordinate (as expected by correspondences)
                targets_sorted = sorted(targets, key=lambda t: t.x)
                frm.targets[cam - 1][:len(targets_sorted)] = targets_sorted
                frm.num_targets[cam - 1] = len(targets_sorted)
                if len(targets_sorted) > 0:
                    arr = np.recarray(len(targets_sorted), dtype=[('x', 'f8'), ('y', 'f8'), ('pnr', 'i4')])
                    for i, t in enumerate(targets_sorted):
                        arr[i].x = t.x
                        arr[i].y = t.y
                        arr[i].pnr = t.pnr
                else:
                    arr = np.recarray(0, dtype=[('x', 'f8'), ('y', 'f8'), ('pnr', 'i4')])
                corrected.append(arr)
            print(f"[DEBUG] frm.num_targets for frame {frame}: {frm.num_targets}")
            if frame == seq_par.first:
                for cam in range(4):
                    print(f"[DEBUG] corrected[{cam}] for frame {frame}: {corrected[cam]}")
                    print(f"[DEBUG] corrected[{cam}] pnr for frame {frame}: {[corrected[cam][i].pnr for i in range(len(corrected[cam]))]}")
            match_counts = [0] * 5
            con = correspondences(frm, corrected, vpar, cpar, cal, match_counts)
            print(f"[DEBUG] match_counts for frame {frame}: {match_counts}")
            num_parts = match_counts[3]
            print(f"[DEBUG] num_parts for frame {frame}: {num_parts}")
            print(f"[DEBUG] valid correspondences for frame {frame}:")
            cor_buf = np.recarray(num_parts, dtype=Corres_dtype)
            path_buf = []
            for i in range(num_parts):
                cor = con[i]
                print(f"  corresp {i}: p={cor.p}, corr={cor.corr}")
                cor_buf[i].nr = i + 1
                cor_buf[i].p = cor.p
                # Triangulate using point_position
                img_pts = []
                for cam in range(4):
                    idx = cor.p[cam]
                    if idx >= 0:
                        t = frm.targets[cam][idx]
                        x_metric, y_metric = pixel_to_metric(t.x, t.y, cpar)
                        img_pts.append([x_metric, y_metric])
                    else:
                        img_pts.append([COORD_UNUSED, COORD_UNUSED])
                img_pts = np.array(img_pts)
                print(f"    img_pts: {img_pts}")
                _, X = point_position(img_pts, 4, mm, cal)
                print(f"    triangulated X: {X}")
                pi = Pathinfo()
                pi.x = X
                path_buf.append(pi)
            from openptv_python.tracking_frame_buf import write_path_frame
            write_path_frame(cor_buf[:num_parts], path_buf[:num_parts], num_parts, corres_file_base, linkage_file_base, prio_file_base, frame)
            ref_file = src_dir / "res_orig" / f"rt_is.{frame}"
            gen_file = res_dir / f"rt_is.{frame}"
            with open(ref_file, "r") as f:
                ref_lines = f.readlines()
            with open(gen_file, "r") as f:
                gen_lines = f.readlines()
            print(f"[DEBUG] Reference rt_is.{frame} (first 10 lines):\n" + "".join(ref_lines[:10]))
            print(f"[DEBUG] Generated rt_is.{frame} (first 10 lines):\n" + "".join(gen_lines[:10]))
            if not filecmp.cmp(str(ref_file), str(gen_file), shallow=False):
                print(f"[DEBUG] Mismatch in rt_is file for frame {frame}")
                assert False, f"Mismatch in rt_is file for frame {frame}"

        # 3. Tracking
        buf_len = 4
        max_targets = 1000
        flatten_tol = 0.0
        run = TrackingRun(
            seq_par=seq_par,
            tpar=tpar,
            vpar=vpar,
            cpar=cpar,
            buf_len=buf_len,
            max_targets=max_targets,
            corres_file_base=corres_file_base,
            linkage_file_base=linkage_file_base,
            prio_file_base=prio_file_base,
            cal=cal,
            flatten_tol=flatten_tol,
        )
        print("[DEBUG] Running tracking pipeline...")
        track_forward_start(run)
        for step in range(seq_par.first + 3, seq_par.last + 1):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, seq_par.last)
        print("[DEBUG] Tracking pipeline complete.")

        # Print 3D trajectories for each frame
        print("[TRAJECTORIES] 3D positions by frame:")
        for i, frame in enumerate(run.fb.buf):
            pos = frame.positions()
            print(f"Frame {i}: {pos}")

        # 4. Compare generated target files to reference (optional)
        for cam in range(1, 5):
            for frame in range(10001, 10006):
                target_file = img_dir / f"cam{cam}.{frame}_targets"
                assert target_file.exists(), f"Missing target file: {target_file}"
        print("[DEBUG] Test completed for burgers workflow.")
