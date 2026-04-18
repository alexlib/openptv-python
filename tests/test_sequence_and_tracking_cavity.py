import shutil

import shutil
from pathlib import Path
import tempfile
import pytest

from openptv_python.parameters import (
    read_sequence_par, read_control_par, read_track_par, read_volume_par, convert_track_par_to_tuple
)
from openptv_python.tracking_run import tr_new
from openptv_python.track import track_forward_start, trackcorr_c_loop, trackcorr_c_finish
from openptv_python.calibration import read_ori


@pytest.mark.integration
def test_sequence_and_tracking_on_cavity():
    src_dir = Path("tests/testing_folder/test_cavity")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        work_dir = tmp_path / "test_cavity"
        shutil.copytree(src_dir, work_dir)

        param_dir = work_dir / "parameters"
        cal_dir = work_dir / "cal"
        img_orig_dir = work_dir / "img_orig"
        img_dir = work_dir / "img"
        res_dir = work_dir / "res_test"

        # Copy img_orig to img for the test
        shutil.copytree(img_orig_dir, img_dir)
        res_dir.mkdir(exist_ok=True)

        # Copy correspondence files from res_orig to res_test
        res_orig_dir = work_dir / "res_orig"
        for f in res_orig_dir.glob("rt_is.*"):
            shutil.copy(f, res_dir / f.name)
    src_dir = Path("tests/testing_folder/test_cavity")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        work_dir = tmp_path / "test_cavity"
        shutil.copytree(src_dir, work_dir)


        param_dir = work_dir / "parameters"
        cal_dir = work_dir / "cal"
        img_orig_dir = work_dir / "img_orig"
        img_dir = work_dir / "img"
        res_dir = work_dir / "res_test"

        # Copy img_orig to img for the test
        shutil.copytree(img_orig_dir, img_dir)
        res_dir.mkdir(exist_ok=True)

        # Patch sequence.par and ptv.par to use img/ instead of img_orig/ (ensure correct directory)
        seq_par_file = param_dir / "sequence.par"
        ptv_par_file = param_dir / "ptv.par"
        # Patch sequence.par
        with open(seq_par_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(4):
            if lines[i].startswith("img_orig/"):
                lines[i] = lines[i].replace("img_orig/", "img/")
        with open(seq_par_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        # Patch ptv.par
        with open(ptv_par_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(1, 9, 2):
            if lines[i].startswith("img_orig/"):
                lines[i] = lines[i].replace("img_orig/", "img/")
        with open(ptv_par_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        param_dir = work_dir / "parameters"
        cal_dir = work_dir / "cal"
        img_dir = work_dir / "img_orig"
        res_dir = work_dir / "res_test"

        # Load parameters
        seq_par = read_sequence_par(param_dir / "sequence.par", num_cams=4)
        cpar = read_control_par(param_dir / "ptv.par")
        tpar = convert_track_par_to_tuple(read_track_par(param_dir / "track.par"))

        # Create a minimal dummy VolumePar (not used in this test case)
        from openptv_python.parameters import VolumePar
        vpar = VolumePar(
            x_lay=[0.0, 1.0],
            z_min_lay=[0.0, 0.0],
            z_max_lay=[1.0, 1.0],
            cn=0.0, cnx=0.0, cny=0.0, csumg=0.0, eps0=0.0, corrmin=0.0
        )

        # Load calibrations for all 4 cameras
        cal = []
        for cam in range(1, 5):
            ori_file = cal_dir / f"cam{cam}.tif.ori"
            add_file = cal_dir / f"cam{cam}.tif.addpar"
            cal.append(read_ori(ori_file, add_file))

        # File base names for results
        corres_file_base = str(res_dir / "rt_is")
        linkage_file_base = str(res_dir / "ptv_is")
        prio_file_base = str(res_dir / "added")

        # Create results directory
        res_dir.mkdir(exist_ok=True)

        # TrackingRun buffer length and max_targets (use reasonable defaults)
        buf_len = 10
        max_targets = 1000
        flatten_tol = 0.0

        # Create TrackingRun

        # Manually construct TrackingRun with dummy vpar
        from openptv_python.tracking_run import TrackingRun
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

        # Run tracking steps
        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last + 1):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)


        # Check that results were written
        assert res_dir.exists(), "Results directory not created"
        assert any(res_dir.glob("*")), "No result files created"
        print(f"Tracking completed for test_cavity in {work_dir}")

        # Clean up img and res_test folders after test
        shutil.rmtree(img_dir)
        shutil.rmtree(res_dir)
