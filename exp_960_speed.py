import pandas

from detect import run
from datetime import datetime
import pandas as pd
from pathlib import Path

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l",
          "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"]

PRECISION = ["fp16"]

MODEL_TYPES = ["torch", "onnx", "openvino"]


def run_exp_960_speed(models, precisions, model_types):
    column_names = ["model_type", "model", "precision", "prep_time", "NMS_time", "latency", "inference_time",
                    "total_time",
                    "FPS", "experiment_time"]
    map_fps = pd.DataFrame(columns=column_names)

    counter = 0
    aggr_processing_times = []
    for model_type in model_types:
        for model in models:
            imgsize = 960
            for precision in precisions:
                if precision == 'int8' and model == 'yolov5l6':
                    break
                start_experiment = datetime.now()
                row = [model_type, model, precision]
                print(row)
                if model_type == 'openvino':
                    model_path = Path(f'./openvino_models/{model}_{imgsize}_{precision}')
                elif model_type == 'onnx':
                    model_path = Path(f'./onnx_models/{model}_{imgsize}.onnx')
                elif model_type == 'torch':
                    model_path = f'{model}.pt'
                else:
                    print('Not a valid model type')
                print(model_path)

                temp, processing_times = run(
                    weights=model_path,
                    source="../datasets/coco/images/val2017",  # /datasets/coco128/images/train2017/
                    nosave=True,
                    imgsz=(imgsize, imgsize)
                )
                aggr_processing_times.extend(processing_times)

                # create row for map_fps data
                row.append(temp[0])  # preprocessing
                row.append(temp[2])  # NMS
                row.append(temp[0] + temp[2])  # latency (prep + NMS)
                row.append(temp[1])  # inference
                row.append(sum(temp))  # total time
                row.append(1 / (sum(temp)) * 1E3)  # FPS
                row.append((datetime.now() - start_experiment).seconds)  # duration of experiment
                map_fps.loc[counter] = row
                counter += 1
    # store inference times of all images per model
    filename = Path(
        f'results/experiments/exp_960_fps/{datetime.now().strftime("%y%m%d-%H-%M")}_processing_times')
    filename.parent.mkdir(parents=True, exist_ok=True)
    df_aggr_processing_times = pd.DataFrame(aggr_processing_times,
                                            columns=['model_ext', 'model', 'path_dets', 'prep_time', 'normalize',
                                                     'inference', 'nms'])
    df_aggr_processing_times.round(3)
    df_aggr_processing_times.to_csv(str(filename) + '.csv')

    # store mAP results
    filename = Path(f'results/experiments/exp_960_fps/{datetime.now().strftime("%y%m%d-%H-%M")}_exp_time')
    filename.parent.mkdir(parents=True, exist_ok=True)
    map_fps.round(3)
    # x.to_pickle(str(filename) + '.pkl')
    map_fps.to_csv(str(filename) + '.csv')


# #%%
# import os
#
# path = './onnx_models/yolov5l6_640.onnx'
# split = os.path.split(path)
# model = os.path.splitext(split[1])[0]
# model_extension = os.path.splitext(split[1])[1].replace('.', '')
# print(f'model:'
#       f' {model} model_extension: {model_extension}')


#
# import pandas as pd
#
# file = pd.read_pickle(
#     "./results/experiments/exp_960_speed/220623-Jun-06_speed.pkl")
# file.to_csv("./results/experiments/exp_960_speed/220623-Jun-06_speed.csv")
# #%%
if __name__ == "__main__":
    run_exp_960_speed(MODELS, ['fp32'], MODEL_TYPES)
    run_exp_960_speed(MODELS, ['fp32', 'fp16', 'int8'], ['openvino'])
