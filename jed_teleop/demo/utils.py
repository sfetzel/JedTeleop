import argparse

from jed_teleop import MediaPipeHandPoseEstimator
from jed_teleop.sources import RealSenseSource
from jed_teleop.sources.OpenCvDepthEstSource import OpenCvDepthEstSource


def get_demo_estimator():
    parser = argparse.ArgumentParser(
        prog='JedTeleop',
        description='Shows a target box in blue and a red box corresponding to the recognized hand pose.'
                    'Try to bring the red box to the blue box. Bring your fingers and thumb together to read the distance.'
                    'If the distance is small enough the target box will move to a new location.')
    parser.add_argument('--opencv_device', default="0")
    parser.add_argument('--realsense', action="store_true", help="Use a realsense camera using pyrealsense2")
    parser.add_argument("--relative", action="store_true", help="Use relative movements instead of absolute values")
    parser.add_argument("--stretch", type=float, default=1.5, help="Scale for location transformation.")
    parser.add_argument("--finger-distance-threshold", default=0.07,
                        help="Consider gripper closed when fingers and thumb distance is smaller than this threshold.")

    args = parser.parse_args()

    if args.realsense:
        source = RealSenseSource()
    else:
        source = OpenCvDepthEstSource(int(args.opencv_device) if args.opencv_device.isnumeric() else args.opencv_device)
    estimator = MediaPipeHandPoseEstimator(source, [args.stretch, args.stretch, args.stretch])
    estimator.finger_distance_threshold = args.finger_distance_threshold
    return estimator, args