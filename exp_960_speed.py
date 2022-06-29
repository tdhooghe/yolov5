from detect import run
from datetime import datetime
import pandas as pd
from pathlib import Path

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l",
          "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"]

PRECISION = ["fp16"]


def run_exp_960_speed(models, precisions, model_type='torch'):
    column_names = ["model", "precision", "prep_time", "NMS_time", "latency", "inference_time",
                    "total_time",
                    "FPS", "experiment_time"]
    map_fps = pd.DataFrame(columns=column_names)

    counter = 0
    processing_times = []
    for model in models:
        imgsize = 960
        for precision in precisions:
            start_experiment = datetime.now()
            row = [model, precision]
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
                source="../datasets/coco/images/val2017/000000463199.jpg",  # 000000463199.jpg
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
            map_fps.loc[counter] = row
            counter += 1
    print(map_fps)
    # store mAP results
    filename = Path(f'results/experiments/exp_960_speed/{datetime.now().strftime("%y%m%d-%H-%M")}_map_{model}_'
                    f'{model_type}')
    filename.parent.mkdir(parents=True, exist_ok=True)
    map_fps.round(3)
    #x.to_pickle(str(filename) + '.pkl')
    map_fps.to_csv(str(filename) + '.csv')

    # store inference times of all images
    filename = Path(f'results/experiments/exp_960_speed/{datetime.now().strftime("%y%m%d-%H-%M")}_processing_times_'
                    f'{model}_{model_type}')
    filename.parent.mkdir(parents=True, exist_ok=True)

    df_processing_times = pd.DataFrame(processing_times, columns=['path_dets', 'prep_time', 'normalize', 'inference',
                                                                  'nms'])
    df_processing_times.round(3)
    #x.to_pickle(str(filename) + '.pkl')
    df_processing_times.to_csv(str(filename) + '.csv')


if __name__ == "__main__":
    run_exp_960_speed(['yolov5n'], PRECISION, model_type='openvino')
