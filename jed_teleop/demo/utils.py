import argparse

from jed_teleop import MediaPipePoseEstimator
from jed_teleop.hamer_pose_estimator import HamerPoseEstimator
from jed_teleop.sources import RealSenseSource
from jed_teleop.sources.OpenCvDepthEstSource import OpenCvDepthEstSource
from jed_teleop.utils import BufferlessCapture


def get_demo_estimator():
    parser = argparse.ArgumentParser(
        prog='JedTeleop',
        description='Shows a target box in blue and a red box corresponding to the recognized hand pose.'
                    'Try to bring the red box to the blue box. Bring your fingers and thumb together to read the distance.'
                    'If the distance is small enough the target box will move to a new location.')
    parser.add_argument('--opencv_device', default="0")
    parser.add_argument('--type', default="mediapipe", choices=["mediapipe", "realsense", "hamer"],
                        help="mediapipe: Use mediapipe with DepthAnythingV2, realsense: use Mediapipe with realsense camera, hamer: use hamer model")
    parser.add_argument("--relative", action="store_true", help="Use relative movements instead of absolute values")
    parser.add_argument("--stretch", type=float, default=1.5, help="Scale for location transformation.")
    parser.add_argument("--finger-distance-threshold", default=0.07,
                        help="Consider gripper closed when fingers and thumb distance is smaller than this threshold.")

    args = parser.parse_args()
    opencv_specifier = int(args.opencv_device) if args.opencv_device.isnumeric() else args.opencv_device

    if args.type == "realsense":
        source = RealSenseSource()
    elif args.type == "hamer":
        source = BufferlessCapture(opencv_specifier)
    else:
        source = OpenCvDepthEstSource(opencv_specifier)

    if args.type == "hamer":
        estimator = HamerPoseEstimator(source)
    else:
        estimator = MediaPipePoseEstimator(source, [args.stretch, args.stretch, args.stretch])
    estimator.finger_distance_threshold = args.finger_distance_threshold
    return estimator, args