from detect import run

MODEL = ["yolov5n.pt"]
MODELS = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5n6.pt", "yolov5s6.pt",
          "yolov5m6.pt", "yolov5l6.pt"]
IMGSIZE = [""]

FP16bools = [False, True]
experiment2_results = {}


def run_experiment2():
    for model in MODELS:
        experiment2_results[model] = {}
        for FP16bool in FP16bools:
            experiment2_results[model][FP16bool] = run(weights=model,
                source="C:/Users/104JVE/PycharmProjects/datasets/coco/images/test/",
                half=FP16bool,
                nosave=True,
                imgsz=)
    print(experiment2_results)

if __name__ == "__main__":
    run_experiment2()
