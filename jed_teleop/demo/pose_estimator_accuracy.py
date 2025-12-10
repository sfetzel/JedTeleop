import argparse
import random
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from ..pose_estimator import MockEstimator, CircleEstimator, RotatorEstimator, GripperState
from ..hand_pose_estimator import HandPoseEstimator

parser = argparse.ArgumentParser(
                    prog='LacnyTeleop',
                    description='Shows a target box in blue and a red box corresponding to the recognized hand pose.'
                                'Try to bring the red box to the blue box. Bring your fingers and thumb together to read the distance.'
                                'If the distance is small enough the target box will move to a new location.')
parser.add_argument('--opencv_device', default="0")
parser.add_argument("--relative", action="store_true", help="Use relative movements instead of absolute values")
parser.add_argument("--finger-distance-threshold", default=0.07, help="Consider gripper closed when fingers and thumb distance is smaller than this threshold.")

args = parser.parse_args()
estimator = HandPoseEstimator(int(args.opencv_device) if args.opencv_device.isnumeric() else args.opencv_device)
estimator.finger_distance_threshold = args.finger_distance_threshold
#estimator = RotatorEstimator(np.array([0.0,0.0,0.1]))
#estimator = CircleEstimator()
estimator.start()

relative_mode = args.relative

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
visualizer.add_geometry(mesh_frame)

box_size = 0.1
target_box :o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh().create_box(box_size*0.9, box_size*0.9, box_size*0.9)
target_box.translate(np.array([-box_size / 2, -box_size / 2, -box_size / 2]), relative=True)
target_pos = np.array([0.25, 0.25, 0])
target_box.translate(target_pos, relative=False)
target_box.paint_uniform_color(np.array([1.0, 0, 0]))

target_box.rotate(target_box.get_rotation_matrix_from_xyz([0, 55.0, 0]))

pose_box = o3d.geometry.TriangleMesh().create_box(box_size, box_size, box_size)
pose_box.paint_uniform_color(np.array([0.0, 0, 1.0]))
pose_box_position = np.array([0.4, 0.1, 0.0])
pose_box.translate(pose_box_position, relative=False)
prev_rotation = None
last_gripper_state = None

def update_box(vis) -> bool:
    global prev_rotation, target_pos, last_gripper_state, pose_box_position
    gripper_state = None
    if relative_mode:
        delta = estimator.get_deltas()
        if delta is not None:
            pose_box.translate(delta[:3], relative=True)
            rot = R.from_euler('xyz', delta[3:6], degrees=False)
            rot_matrix = rot.as_matrix()
            pose_box.rotate(rot_matrix)
            gripper_state = delta[-1]
            pose_box_position += delta[:3]
    else:
        target = estimator.current_position
        if target is not None:
            pose_box.translate(target[:3], relative=False)
            if prev_rotation is not None:
                pose_box.rotate(np.linalg.inv(prev_rotation))
            rot = R.from_euler('xyz', target[3:6], degrees=False)
            rot_matrix = rot.as_matrix()
            pose_box.rotate(rot_matrix)
            prev_rotation = rot_matrix
            gripper_state = target[-1]
            pose_box_position = target[:3]
    vis.update_geometry(pose_box)

    if gripper_state is not None:
        if gripper_state != last_gripper_state:
            print(f"Gripper is: {"Closed" if gripper_state == GripperState.Closed.value else "Open"}")
        last_gripper_state = gripper_state

        if gripper_state == GripperState.Closed.value:
            distance = np.linalg.norm(pose_box_position - target_pos)
            if distance < 0.2:
                print(f"Distance to target: {distance}")
            if distance < 0.04:
                print(f"Well done, distance to target: {distance}")
                target_pos = np.array([random.uniform(0.1, 0.5),
                                       random.uniform(0.1, 0.5),
                                       random.uniform(0.1, 0.25)])
                target_box.rotate(target_box.get_rotation_matrix_from_xyz(np.random.uniform(0, 50.0, 3)))
                target_box.translate(target_pos, relative=False)

    time.sleep(0.1)
    return True


visualizer.add_geometry(target_box)
visualizer.add_geometry(pose_box)
visualizer.register_animation_callback(update_box)
visualizer.set_view_status('{"class_name":"ViewTrajectory","interval":29,"is_loop":false,"trajectory":[{"boundingbox_max":[1,1,1],"boundingbox_min":[-0.06,-0.06,-0.06],"field_of_view":60,"front":[0.041777246550877556,-0.8522171023054014,0.5215176624130886],"lookat":[0.49897655892683684,0.3679501731479167,0.30091815496885477],"up":[-0.08145278023458108,0.5173308649632169,0.8519003584624595],"zoom":0.7}],"version_major":1,"version_minor":0}')
visualizer.poll_events()
visualizer.update_renderer()
visualizer.run()
estimator.stop()