from export import run

MODELS = ["yolov5n", #"yolov5s", "yolov5m", "yolov5l",
          # "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"
          ]

PRECISION = ['fp16',
             # 'fp32'
             ]
# try different images sizes, 640 not necessary as already pretrained on 640
IMAGE_SIZES = [320,
#480, 960
]


def export_models(models, precision, image_sizes):
    # export models to openvino format
    for model in models:
        for precision in precision:
            for image_size in image_sizes:
                weights = model + '.pt'
                run(weights=weights, imgsz=(image_size, image_size), ov_fp16=precision, include=('onnx', 'openvino'), simplify=True)


if __name__ == "__main__":
    export_models(MODELS, PRECISION, IMAGE_SIZES)
