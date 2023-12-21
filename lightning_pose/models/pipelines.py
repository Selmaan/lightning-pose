import torch
from lightning_pose.models.heatmap_tracker import HeatmapTracker
from omegaconf import DictConfig
from typing import Tuple, Optional, Union
from torchtyping import TensorType
import torchvision.transforms.v2 as tchv2
from lightning_pose.utils.predictions import load_model_from_checkpoint
from lightning_pose.data.utils import HeatmapLabeledBatchDict, UnlabeledBatchDict
from lightning_pose.models.base import convert_bbox_coords

class DynamicPipeline(HeatmapTracker):
    """DynamicPipeline is a HeatMapTracker subclass, which takes standard full-frame batch_dict inputs
    and handles detection and crop/resize of image, keypoints, and bbox before passing new
    'prepared' batch_dict to a standard HeatmapTracker"""

    def __init__(self,
                 cfg: DictConfig,
                 loss_factory: Optional[LossFactory] = None,
                 torch_seed: int = 123,
                 lr_scheduler: str = "multisteplr",
                 lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        ) -> None:
        super().__init__(
            num_keypoints=cfg.data.num_keypoints,
            loss_factory=loss_factory,
            backbone=cfg.model.backbone,
            downsample_factor = cfg.data.downsample_factor,
            pretrained=cfg.model.get("backbone_pretrained", True),
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params = lr_scheduler_params,
        )
        print(f"\n Initializing a Dynamic Pipeline around {self._get_name()} instance.")

        # set up resizing operations for COM and POS networks
        self.resized_dims = (cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width)
        self.posture_resize = tchv2.Resize(self.resized_dims, antialias=True)
        self.com_resized_dims = (cfg.detector.image_resize_dims.height, cfg.detector.image_resize_dims.width)
        self.com_resize = tchv2.Resize(self.com_resized_dims, antialias=True)

        # set up output shape from resize dim TODO: what is this used for? Is this necessary?
        self.output_shape = (self.resized_dims[0] // 2**self.downsample_factor,
                             self.resized_dims[1] // 2**self.downsample_factor)

        # load pre-trained detector network: TODO: is this an issue if POS and COM configs differ?
        self.detector_model = load_model_from_checkpoint(cfg=cfg, ckpt_file=cfg.detector.ckpt_path,
                                                         skip_data_module = True, eval=True).to('cuda')
        # store keypoint index for COM
        self.keypoint_for_com = cfg.detector.keypoint_for_com

        if not cfg.data.dynamic_crop:
            print("Dynamic Pipeline should not be used without dynamic crop config")
            exit()

    def detect_batch(
            self,
            batch_dict: HeatmapLabeledBatchDict,
        ) -> TensorType["batch", "num_targets"]:

        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]
        with torch.no_grad():
            # downsize images
            ds_images = self.detector_resize(images)
            # images -> heatmaps
            predicted_heatmaps = self.detector_model(ds_images)
            # heatmaps -> keypoints
            predicted_keypoints, confidence = self.detector_model.run_subpixelmaxima(predicted_heatmaps)
            # convert from downsized to original pixel coordinates
            # TODO: This breaks when image is augmented by resizing/cropping/rotating!
            predicted_keypoints = convert_bbox_coords(batch_dict, predicted_keypoints)
        return predicted_keypoints

    def crop_batch(
            self,
            batch_dict: HeatmapLabeledBatchDict,
            centroids: TensorType["batch", 2],
            crop_size: TensorType["batch", 1],
        ) -> HeatmapLabeledBatchDict:
        """transform images, bbox, and keypoints if present from full to cropped"""
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]
        # create new dict, initialize cropped image array
        crop_dict = batch_dict.copy()
        crop_ims = torch.zeros_like(images)
        # loop over images
        for i in range(images.shape[0]):
            # crop (padding w/ zeros) and resize each image using centroid and size data
            crop_ims[i] = self.posture_resize(tchv2.functional.crop(images[i],
                centroids[i,1]-crop_size[i], centroids[i,0]-crop_size[i], 2*crop_size[i], 2*crop_size[i]))  # y,x,h,w
        # add new image array to crop dictionary
        if "images" in crop_dict.keys():
            # labeled image dataloaders
            crop_dict["images"] = crop_ims
        else:
            # unlabeled dali video dataloaders
            crop_dict["frames"] = crop_ims
        # update bbox info
        crop_dict['bbox'][:,0] = centroids[:,0]-crop_size
        crop_dict['bbox'][:,1] = centroids[:,1]-crop_size
        crop_dict['bbox'][:,2] = 2*crop_size
        crop_dict['bbox'][:,3] = 2*crop_size
        # update keypoints if present
        if "keypoints" in crop_dict.keys():
            # convert keypoints to x,y format
            num_targets = crop_dict['keypoints'].shape[1]
            num_keypoints = num_targets // 2
            kp = crop_dict['keypoints'].reshape((-1, num_keypoints, 2))
            # convert x,y coords from original to cropped coords
            kp[:, :, 0] -= crop_dict['bbox'][:,0].unsqueeze(1)  # remove x offset
            kp[:, :, 0] *= self.resized_dims[1] / crop_dict['bbox'][:, 3].unsqueeze(1)  # rescale x coords by width
            kp[:, :, 1] -= crop_dict['bbox'][:,1].unsqueeze(1)  # remove y offset
            kp[:, :, 1] *= self.resized_dims[0] / crop_dict['bbox'][:, 2].unsqueeze(1)  # rescale y coords by height
        return crop_dict

    def get_crop_params(
            self,
            predicted_keypoints: TensorType["batch", "num_targets"],
            fill_value = 200,
        ) -> tuple:
        # get COM coordinates in x,y format
        num_targets = predicted_keypoints.shape[1]
        num_keypoints = num_targets // 2
        predicted_keypoints = predicted_keypoints.reshape((-1, num_keypoints, 2))
        centroids = predicted_keypoints[:, self.keypoint_for_com].round().to(torch.int)
        # set crop size for each image, TODO: extend crop size functionality beyond this test heuristic
        crop_size = torch.full((centroids.shape[0], 1),fill_value)
        return centroids, crop_size  # (nBatch x 2, nBatch x 1)

    def prepare_batch(self, batch_dict):
        # get all detector predictions
        predicted_keypoints = self.detect_batch(batch_dict)
        centroids, crop_size = self.get_crop_params(predicted_keypoints)
        # crop images and update bbox, also keypoints if applicable
        crop_dict = self.crop_batch(batch_dict, centroids, crop_size)
        return crop_dict

    def get_loss_inputs_labeled(
        self,
        batch_dict: Union[
            HeatmapLabeledBatchDict,
        ],
    ) -> dict:
        # pre-process batch_dict
        crop_dict = self.prepare_batch(batch_dict)
        # call standard method on cropped dict
        return super().get_loss_inputs_labeled(crop_dict)

    def predict_step(
        self,
        batch_dict: Union[HeatmapLabeledBatchDict, UnlabeledBatchDict],
        batch_idx: int,
        return_heatmaps: Optional[bool] = False,
        ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]:
        # pre-process batch_dict
        crop_dict = self.prepare_batch(batch_dict)
        # call standard method on cropped dict
        return super().predict_step(batch_dict=crop_dict, batch_idx=batch_idx, return_heatmaps=return_heatmaps)