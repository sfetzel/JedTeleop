# JedTeleop
Simple teleoperation via webcam using hand pose recognition from MediaPipe and monocular depth estimation from DepthAnythingV2

![](teleop_demo_short.gif)

## Install
- `pip install -r requirements.txt`
- `pip install -e .`

## Try it out
- `python -m jed_teleop.demo.pose_estimator_accuracy` for an open3d visualization (from screenshot).
- `python -m jed_teleop.demo.pose_estimator_vis` for a matplotlib visualization.

## Controls
For opencv image window:
- Press "q" to close the window and stop the estimator thread.
- Press "p" to pause the calculation of delta positions.
- Press "z" to set the current position to the origin position.
- Press "c" to set the current orientation of the hand normal to be equal to z-axis.