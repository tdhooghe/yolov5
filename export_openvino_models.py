from export import run

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5n6", "yolov5s6",
          "yolov5m6", "yolov5l6"]

PRECISION = ['fp16', 'fp32']


def export_models():
    # export models to openvino format
    for model in MODELS:
        for precision in PRECISION:
            imgsize = 1280 if '6' in model else 640
            weights = model + '.pt'
            run(weights=weights, imgsz=(imgsize, imgsize), ov_fp16=precision, include=('onnx', 'openvino'))


if __name__ == "__main__":
    export_models()
