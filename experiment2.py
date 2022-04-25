import pandas

from detect import run
from datetime import datetime
import pandas as pd


# python export.py --weights yolov5n.pt yolov5s.pt yolov5m.pt yolov5l.pt yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt --include openvino
MODEL = ["yolov5n_openvino_model/yolov5n.xml"]
MODELS_OPENVINO = ["yolov5n_openvino_model/yolov5n.xml",
                   "yolov5s_openvino_model/yolov5s.xml",
                   "yolov5m_openvino_model/yolov5m.xml",
                   "yolov5l_openvino_model/yolov5l.xml",
                   "yolov5n6_openvino_model/yolov5n6.xml",
                   "yolov5s6_openvino_model/yolov5s6.xml",
                   "yolov5m6_openvino_model/yolov5m6.xml",
                   "yolov5l6_openvino_model/yolov5l6.xml"]
MODELS = ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5n6.pt", "yolov5s6.pt",
          "yolov5m6.pt", "yolov5l6.pt"]

# BATCH_SIZES = [1, 16, 32]
# FP16bools = [True]

column_names = ["model", "img_size", "prep_time", "inference_time", "NMS_time", "total_time",
                "latency", "FPS", "experiment_time"]
exp2_df_results = pd.DataFrame(columns=column_names)


def run_experiment2():
    counter = 0
    for model in MODELS_OPENVINO:
        # experiment2_results[model] = {}
        imgsize = 1280 if '6' in model else 640
        # for FP16bool in FP16bools:
        #     precision = 'FP16' if FP16bool else 'FP32'
        #     for batch_size in BATCH_SIZES:
        start_experiment = datetime.now()
        row = [model, imgsize]
        print(row)
        temp = run(
            weights=model,
            source="C:/Users/104JVE/PycharmProjects/datasets/coco/images/exp2/",
            # half=FP16bool,
            imgsz=(imgsize, imgsize),
            nosave=True,
        )
        for i in temp:
            row.append(i)
        row.append(sum(temp))  # total time
        row.append(temp[0] + temp[2])  # latency (prep + NMS)
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
    run_experiment2()
