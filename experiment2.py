from detect import run

MODEL = ["yolov5n.pt"]
MODELS = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5n6.pt", "yolov5s6.pt",
          "yolov5m6.pt", "yolov5l6.pt"]

PRECISION = [False, True]

def run_experiment2():
    for model in MODEL:
        for precision in PRECISION:
            run(weights=model,
                source="C:/Users/104JVE/PycharmProjects/datasets/coco/images/test/1.jpg",
                half=PRECISION,
                nosave=True)


if __name__ == "__main__":
    run_experiment2()
