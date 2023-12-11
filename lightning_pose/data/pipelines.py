import torch
from lightning_pose.data.datasets import BaseTrackingDataset
from lightning_pose.utils.predictions import load_model_from_checkpoint
import torchvision.transforms.v2 as tchv2
from omegaconf import DictConfig
from typing import Optional, List, Callable
from lightning_pose.data.utils import (
    BaseLabeledExampleDict,
    HeatmapLabeledExampleDict,
    MultiviewHeatmapLabeledExampleDict,
    generate_heatmaps,
)

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DynamicPipeline(BaseTrackingDataset):
    """Dynamic dataset used to load data for dynamic pipeline"""

    def __init__(
        self,
        cfg: DictConfig,
        imgaug_transform: Optional[Callable] = None,
        header_rows: Optional[List[int]] = [0, 1, 2],
        do_context: bool = False,
        uniform_heatmaps: bool = False,
    ) -> None:
        super().__init__(
            root_directory=cfg.data.data_dir,
            csv_path=cfg.data.csv_file,
            header_rows=header_rows,
            imgaug_transform=imgaug_transform,
            do_context=do_context,
        )
        self.resized_dims = (cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width)
        self.downsample_factor = cfg.data.downsample_factor
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor
        self.uniform_heatmaps = uniform_heatmaps
        self.num_targets = torch.numel(self.keypoints[0])
        self.num_keypoints = self.num_targets // 2
        if self.height % 128 != 0 or self.height % 128 != 0:
            print(
                "image dimensions (after transformation) must be repeatably "
                + "divisible by 2!"
            )
            print("current image dimensions after transformation are:")
            exit()
        if not cfg.data.dynamic_crop:
            print("Dynamic Pipeline should not be used without dynamic crop config")

        self.init_detector(cfg)

    @property
    def height(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.resized_dims[1]

    @property
    def width(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.resized_dims[0]

    @property
    def output_shape(self) -> tuple:
        return (
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    def init_detector(self, cfg) -> None:
        resized_dims = (cfg.detector.image_resize_dims.height, cfg.detector.image_resize_dims.width)
        # load pre-trained detector network: Note this assumes same model class for detector and pose networks!
        self.detector_model = load_model_from_checkpoint(cfg=cfg, ckpt_file=cfg.detector.ckpt_path,
                                                         skip_data_module = True, eval=True).to('cuda')
        # set up resize transform
        self.detector_resize = tchv2.Resize(resized_dims, antialias=True)
        # store keypoint index for COM
        self.keypoint_for_com = cfg.detector.keypoint_for_com


    def __getitem__(self, idx: int) -> HeatmapLabeledExampleDict:
        """Get an example from the dataset.

        Calls the base dataset to get an image and a label, then implements pipeline

        """
        # load full resolution images
        example_dict: BaseLabeledExampleDict = super().__getitem__(idx)
        # transfer images to gpu
        if torch.cuda.is_available():
            example_dict['images'] = example_dict['images'].to('cuda')
        # predict detector keypoints
        with torch.no_grad():
            # downsize images
            ds_images = self.detector_resize(example_dict['images'])
            # images -> heatmaps
            predicted_heatmaps = self.detector_model(ds_images.unsqueeze(0))  # shape as batch size = 1
            # heatmaps -> keypoints
            predicted_keypoints, confidence = self.detector_model.run_subpixelmaxima(predicted_heatmaps)
            # bounding box coords -> original image coords; assume no cropping for detector
            predicted_keypoints = predicted_keypoints.reshape((-1,2))  # x,y format
            predicted_keypoints[:, 0] *= example_dict['bbox'][-1] / ds_images.shape[-1]  # x coords
            predicted_keypoints[:, 1] *= example_dict['bbox'][-2] / ds_images.shape[-2]  # y coords
        # use predicted keypoints to get crop parameters
        crop_size = 200  # heuristic, to be replaced, crop this many px around centroid
        centroid = predicted_keypoints[self.keypoint_for_com].round().to(torch.int)
        # replace image with cropped version (automatically pads image with 0s)
        example_dict['images'] = tchv2.functional.crop(example_dict['images'], centroid[1]-crop_size,
                   centroid[0]-crop_size, 2*crop_size, 2*crop_size)  # y, x, h, w order
        # update image bbox
        example_dict['bbox'] = torch.tensor([centroid[0]-crop_size, centroid[1]-crop_size, 2*crop_size, 2*crop_size])  # x, y, h, w order
        # keypoints and/or heatmaps?
        return example_dict