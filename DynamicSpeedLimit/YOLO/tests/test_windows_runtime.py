"""Exercise the native calibration/PyTorch DLL sequence in a fresh process."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(
    sys.platform == "win32"
    and importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("ultralytics") is not None,
    "Requires Windows with PyTorch and Ultralytics installed",
)
class WindowsRuntimeTests(unittest.TestCase):
    def test_calibration_then_pytorch_without_duplicate_runtime_bypass(self):
        environment = os.environ.copy()
        environment.pop("MKL_THREADING_LAYER", None)
        environment.pop("KMP_DUPLICATE_LIB_OK", None)
        script = """
import sys
sys.path.insert(0, 'YOLO')
import main
assert 'api_code' not in sys.modules
assert 'decision_engine' not in sys.modules
assert 'requests' not in sys.modules
from calibration import RoadCalibration
from tracker import SpeedTracker
calibration = RoadCalibration([[20, 20], [200, 20], [250, 300], [0, 300]], 10, 25)
calibration.perspective_matrix()
tracker = SpeedTracker(calibration.points, 10, 25)
from ultralytics import YOLO
import torch
import numpy as np
assert np.linalg.matrix_rank(np.eye(3)) == 3
assert torch.mm(torch.ones((3, 3)), torch.ones((3, 3))).sum().item() == 27
print('Calibration and PyTorch operations passed')
"""
        result = subprocess.run(
            [sys.executable, "-u", "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Calibration and PyTorch operations passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
