from detect import run
from datetime import datetime
import pandas as pd
import subprocess
from pathlib import Path

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5n6", "yolov5s6",
          "yolov5m6", "yolov5l6"]

MODEL = ["yolov5n_openvino_model"]

# BATCH_SIZES = [1, 16, 32]
PRECISION = ['fp16', 'fp32']


def export_models():
    # create the models
    for model in MODELS:
        for precision in PRECISION:
            path = Path(model + '_openvino_model_' + precision)
            if not path.is_dir():
                imgsize = 1280 if '6' in model else 640
                if precision == 'fp16':
                    cmd = f'python export.py --weights {model} --imgsz {imgsize} --include openvino --ov-fp16'
                else:
                    cmd = f'python export.py --weights {model} --imgsz {imgsize} --include openvino'
                subprocess.check_output(cmd, shell=True)


def run_experiment2():
    column_names = ["model", "img_size", "precision", "prep_time", "NMS_time", "latency", "inference_time",
                    "total_time",
                    "FPS", "experiment_time"]
    exp2_df_results = pd.DataFrame(columns=column_names)

    counter = 0
    for model in MODELS:
        imgsize = 1280 if '6' in model else 640
        for precision in PRECISION:
            # for batch_size in BATCH_SIZES:
            start_experiment = datetime.now()
            row = [model, imgsize, precision]
            print(row)
            temp = run(
                weights=model + '_openvino_model_' + precision,
                source="C:/Users/104JVE/PycharmProjects/datasets/coco/images/exp2/",  # 000000463199.jpg
                imgsz=(imgsize, imgsize),
                nosave=True,
            )
            row.append(temp[0])  # preprocessing
            row.append(temp[2])  # NMS
            row.append(temp[0] + temp[2])  # latency (prep + NMS)
            row.append(temp[1])  # inference
            row.append(sum(temp))  # total time
            row.append(1 / (sum(temp)) * 1E3)  # FPS
            row.append((datetime.now() - start_experiment).seconds)  # duration of experiment
            exp2_df_results.loc[counter] = row
            counter += 1
    print(exp2_df_results)
    # store results
    filename = f'exp2_results/exp2_df_results_{datetime.now().strftime("%d-%m-%Y_%H-%M")}'
    exp2_df_results.to_pickle(filename + '.pkl')
    exp2_df_results.to_csv(filename + '.csv')


if __name__ == "__main__":
    export_models()
    run_experiment2()
