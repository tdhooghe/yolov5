# %%
import os
import time
from pathlib import Path
from typing import Tuple

import cv2

import numpy as np
from compression.api import DataLoader, Metric
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from PIL import Image
from utils.augmentations import letterbox
from openvino.runtime import Core
from utils.general import imread
from yaspin import yaspin
from export_openvino_models import export_models, MODELS

MODELS_P5 = ["yolov5n", "yolov5s", "yolov5m", "yolov5l"]
MODELS_P6 = ["yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"]
MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5n6", "yolov5s6", "yolov5m6"]

class DetectionDataLoader(DataLoader):
    # https://docs.openvino.ai/latest/notebooks/111-detection-quantization-with-output.html#run-quantization-pipeline
    def __init__(self, basedir: str, target_size: Tuple[int, int]):
        """
        :param basedir: Directory that contains images and annotation as "annotation.json"
        :param target_size: Tuple of (width, height) to resize images to.
        """
        self.images = sorted(Path(basedir).glob("*.jpg"))
        self.target_size = target_size

    def __getitem__(self, index):
        """
        Get an item from the dataset at the specified index.
        Detection boxes are converted from absolute coordinates to relative coordinates
        between 0 and 1 by dividing xmin, xmax by image width and ymin, ymax by image height.

        :return: (annotation, input_image, metadata) where annotation is (index, target_annotation)
                 with target_annotation as a dictionary with keys category_id, image_width, image_height
                 and bbox, containing the relative bounding box coordinates [xmin, ymin, xmax, ymax]
                 (with values between 0 and 1) and metadata a dictionary: {"filename": path_to_image}
        """
        image_path = self.images[index]
        image = imread(str(image_path))  # read image with OpenCV
        image, _, _ = letterbox(image, new_shape=self.target_size,
                                auto=False)  # resize to a target input size in nice yolov5 way
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        image = image[:, ::-1, :, :]
        image = np.float32(image)
        image /= 255  # scale values between 0-1, as yolov5 does this as well

        return (
            image,
            None,
            {"filename": str(image_path), "shape": image.shape},
        )

    def __len__(self):
        return len(self.images)


def quantize_models(model_names):
    for model_name in model_names:
        imgsize = 960
        ir_path = Path(f"openvino_models/{model_name}_fp16_{imgsize}/{model_name}_{imgsize}.xml")

        # %%
        model_config = {
            "model_name": ir_path.stem,
            "model": ir_path,
            "weights": ir_path.with_suffix(".bin"),
        }

        # Engine config
        engine_config = {"device": "CPU"}

        default_algorithms = [
            {
                "name": "DefaultQuantization",
                "stat_subset_size": 500,
                "params": {
                    "target_device": "ANY",
                    "preset": "performance",  # choose between "mixed" and "performance"
                },
            }
        ]

        # create data loader
        data_loader = DetectionDataLoader(
            basedir="../datasets/coco/images/quantize", target_size=(imgsize, imgsize)
        )
        ir_model = load_model(model_config=model_config)  # load model from model config
        engine = IEEngine(config=engine_config, data_loader=data_loader)  # initialize engine
        pipeline = create_pipeline(default_algorithms, engine)  # create pipeline of compression algos
        algorithm_name = pipeline.algo_seq[0].name
        with yaspin(
                text=f"Executing POT pipeline on {model_config['model']} with {algorithm_name}"
        ) as sp:
            start_time = time.perf_counter()
            compressed_model = pipeline.run(ir_model)
            end_time = time.perf_counter()
            sp.ok("✔")
        print(f"Quantization finished in {end_time - start_time:.2f} seconds")

        compress_model_weights(compressed_model)  # compress weights to reduce size of .bin file

        compressed_model_paths = save_model(
            model=compressed_model,
            save_path=f"openvino_models/{model_name}_int8_{imgsize}",
            model_name=f"{model_name}",
        )

        compressed_model_path = compressed_model_paths[0]["model"]
        print("The quantized model is stored at", compressed_model_path)


class ImageLoader(DataLoader):
    """ Loads images from a folder """

    def __init__(self, dataset_path):
        # Use OpenCV to gather image files
        # Collect names of image files
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for name in all_files_in_dir:
            file = os.path.join(dataset_path, name)
            if cv2.haveImageReader(file):
                self._files.append(file)

        # Define shape of the model
        self._shape = (640, 640)

    def __len__(self):
        """ Returns the length of the dataset """
        return len(self._files)

    def __getitem__(self, index):
        """ Returns image data by index in the NCHW layout
        Note: model-specific preprocessing is omitted, consider adding it here
        """
        if index >= len(self):
            raise IndexError("Index out of dataset size")
        image = imread(self._files[index])  # read image with OpenCV
        image, _, _ = letterbox(image, auto=False)  # resize to a target input size in nice yolov5 way
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        image = np.float32(image)
        image /= 255
        image = image[:, ::-1, :, :]
        return image, None  # annotation is set to None


# loader = ImageLoader("../datasets/coco/images/exp2")
# for i in range(3):
#     img = loader.__getitem__(i)[0]
#     img = np.squeeze(img)
#     pil_img = np.moveaxis(img, 0, 2)
#     Image.fromarray(pil_img, mode='RGB').show()


if __name__ == "__main__":
    #export_models()
    #quantize_models(MODELS)
    quantize_models(MODELS)
