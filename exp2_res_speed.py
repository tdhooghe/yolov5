from detect import run
from datetime import datetime
import pandas as pd
from pathlib import Path

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l"
          # "yolov5n6", "yolov5s6", "yolov5m6", #"yolov5l6"
          ]

PRECISION = ['fp16']

IMAGE_SIZES = [320, 480, 960]


def run_exp1_speed():
    column_names = ["model", "precision", "prep_time", "NMS_time", "latency", "inference_time",
                    "total_time",
                    "FPS", "experiment_time"]
    exp2_res_speed = pd.DataFrame(columns=column_names)

    counter = 0
    for model in MODELS:
        for precision in PRECISION:
            for imgsize in IMAGE_SIZES:
                start_experiment = datetime.now()
                row = [model, precision]
                print(row)
                model_path = Path(f'./openvino_models/{model}_{precision}_{imgsize}')
                print(model_path)
                temp = run(
                    weights=model_path,
                    source="../datasets/coco/images/val2017",  # 000000463199.jpg
                    nosave=True,
                    imgsz=(imgsize, imgsize)
                )
                row.append(temp[0])  # preprocessing
                row.append(temp[2])  # NMS
                row.append(temp[0] + temp[2])  # latency (prep + NMS)
                row.append(temp[1])  # inference
                row.append(sum(temp))  # total time
                row.append(1 / (sum(temp)) * 1E3)  # FPS
                row.append((datetime.now() - start_experiment).seconds)  # duration of experiment
                print(row)
                exp2_res_speed.loc[counter] = row
                counter += 1
    print(exp2_res_speed)
    # store results
    filename = Path(f'results/experiments/exp2/res_speed_{datetime.now().strftime("%d-%m-%Y_%H-%M")}')
    filename.parent.mkdir(parents=True, exist_ok=True)
    exp2_res_speed.round(3)
    print(exp2_res_speed)
    exp2_res_speed.to_pickle(str(filename) + '.pkl')
    exp2_res_speed.to_csv(str(filename) + '.csv')


if __name__ == "__main__":
    run_exp1_speed()
