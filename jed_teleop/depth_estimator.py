from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

class DepthAnythingEstimator:
    def __init__(self, use_moving_average: bool = True, decay: float = 0.4):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-small-hf", use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-small-hf").to(self.device)
        self.last_depth_image = None
        self.use_moving_average = use_moving_average
        self.decay = decay

    def get_depth(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # interpolate to original size and visualize the prediction
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.shape[0], image.shape[1])],
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())

        depth = depth.detach().cpu().numpy()

        if self.use_moving_average and self.last_depth_image is not None:
            depth = self.decay * self.last_depth_image + (1 - self.decay) * depth

        self.last_depth_image = depth
        return depth
