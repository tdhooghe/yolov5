from export import run

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l"]

MODELS_P6 = ["yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"]

PRECISION = ['fp16']

# try different images sizes, 640 not necessary as already pretrained on 640
IMAGE_SIZES = [320, 480, 640, 800, 960]

IMAGE_SIZES_P6 = [256, 448, 640, 832, 1024]  # image size for P6 models (multiple of stride 64)

IMAGE_SIZES_EXTRA = [256, 384, 512, 640, 768, 896, 1024]


def export_models(models, precisions, image_sizes):
    # export models to openvino format
    for model in models:
        for image_size in image_sizes:
            for precision in precisions:
                half = True if precision == 'fp16' else False
                weights = model + '.pt'
                run(weights=weights, imgsz=(image_size, image_size), half=half,
                    include=('onnx', 'openvino'))


if __name__ == "__main__":
    export_models(MODELS, ['fp32'], [960])
    export_models(MODELS_P6, ['fp32'], [960])
    #export_models(['yolov5n'], ['fp16'], [960])
