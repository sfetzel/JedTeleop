import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from jed_teleop.hands_detection.hands import Handedness
from jed_teleop.hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode
from jed_teleop.keypoint_based_estimator import KeypointBasedEstimator
from jed_teleop.orientation import calculate_normal
from jed_teleop.sources.VideoSource import VideoSource, Frame

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

from jed_teleop.utils import BufferlessCapture, calculate_rotation_matrix

# color for hand mesh
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

class HamerPoseEstimator(KeypointBasedEstimator):

    def __init__(self, source: VideoSource, cache_dir = CACHE_DIR_HAMER, stretch_factors: list = None) -> None:
        """
        Pose Estimator using HaMeR model.
        """
        super(HamerPoseEstimator, self).__init__(source, stretch_factors if stretch_factors is not None else [1.0, 1.0, 0.1])
        self.detector = MediaPipeHandPose(running_mode=VisionRunningMode.VIDEO, min_hand_detected_confidence=0.3,
                                          min_hand_presence_confidence=0.3)
        self.source = source
        self.zero_pos = np.array([0.5, 0.5, 10.0])
        self.horizontal_flip = True
        self.rescale_factor = 2.0 # rescale factor bounding boxes (extend by factor).
        # Download and load checkpoints
        download_models(cache_dir)
        checkpoint_path = f"{cache_dir}/hamer_ckpts/checkpoints/hamer.ckpt"
        self.model, self.model_cfg = load_hamer(checkpoint_path)

        # Setup HaMeR model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.finger_distance_threshold = 0.1
        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    def get_bounding_boxes(self, img):
        h, w, _ = img.shape
        result = self.detector.detect(img)
        bounding_box = None
        right = None

        if result is not None:
            landmarks, handedness = result

            # Bounding box from landmarks
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]

            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            is_right = handedness == Handedness.RIGHT

            bounding_box = [x_min, y_min, x_max, y_max]
            right = 1 if is_right else 0
        return bounding_box, right

    def get_keypoints(self, img_cv2: np.ndarray):
        box, right = self.get_bounding_boxes(img_cv2)
        height, width, _ = img_cv2.shape
        points = []
        result_img = img_cv2
        keypoints_3d = None
        keypoints_2d = None
        cam_t = None
        if box is not None:  # check if any hands detected.
            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(self.model_cfg, img_cv2, np.array([box]), np.array([right]), rescale_factor=self.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            all_verts = []
            all_cam_t = []
            all_right = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                start = time.time()
                with torch.no_grad():
                    out = self.model(batch)
                end = time.time()
                print(f"Inference on hamer took {end - start}s")

                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                multiplier = (2 * batch['right'] - 1)
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                                   scaled_focal_length).detach().cpu().numpy()

                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                                DEFAULT_STD[:, None, None] / 255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                                DEFAULT_MEAN[:, None, None] / 255)
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)
                    # print(out['pred_keypoints_3d'][n])
                    # print(out['pred_cam_t'][n])
                    points.append(torch.tensor(cam_t).to(self.device))
                    keypoints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    # keypoints in range [-0.5,0.5] -> add 0.5 to get [0, 1]
                    keypoints_2d = out['pred_keypoints_2d'][n].detach().cpu().numpy() + 0.5
                    keypoints_2d[:, 0] = (keypoints_2d[:, 0] * box_width + box[0]) / width
                    keypoints_2d[:, 1] = (keypoints_2d[:, 1] * box_height + box[1]) / height


            # Render front view
            if len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n],
                                                         is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[
                    :, :, 3:]
                result_img = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)

        return result_img, cam_t, keypoints_3d, keypoints_2d

    def process_frame(self, img):

        if self.horizontal_flip:
            img = cv2.flip(img, 1)

        img, cam_t, keypoints_3d, keypoints_2d = self.get_keypoints(img)

        if cam_t is not None and keypoints_3d is not None:
            normal = calculate_normal(keypoints_3d)
            self.last_normal = normal
            new_location = cam_t
            new_location[:2] = keypoints_2d[0, :2]
            new_location = self.shift_scale_location(new_location)

            rotation = R.from_matrix(calculate_rotation_matrix(normal))
            new_rotation = rotation.as_euler("xyz")

            self.set_gripper_state(keypoints_3d)

            gripper_value = self.get_gripper_state_int()
            new_pose = np.concatenate([new_location, new_rotation, np.array([gripper_value])])
            print(new_pose)
            self.set_smoothed_pose_and_update_deltas(new_pose)

        return img

if __name__ == "__main__":
    source = BufferlessCapture(0)
    estimator = HamerPoseEstimator(source, "/home/simon/Projects/hamer/_DATA")
    estimator.run()
