from val import run
from datetime import datetime
from exp1_speed import run_exp1_speed
import numpy as np
import pandas as pd


MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l",
          "yolov5n6", "yolov5s6", "yolov5m6",  "yolov5l6"]

PRECISION = ['int8', 'fp16', 'fp32']

# %%
CLASSES = ['person', 'car', 'motorcycle', 'bus', 'truck', 'baseball bat', 'knife', 'cell phone']


def run_exp1_perf(models, precisions):
    # glob_metrics: mp, mr, map50, map
    # t: prep., inference, NMS
    # person: 0, car: 2, motorcycle: 3, bus: 5, truck: 7, baseball bat: 38, knife: 48, cell phone: 76

    column_names = ["model", "img_size", "precision", "mAP50", "mAP", "mAP50_8class", "mAP_8class", "person_mAP.5",
                    "car_mAP.5", "motorcycle_mAP.5", "bus_mAP.5", "truck_mAP.5", "baseball_bat_mAP.5", "knife_mAP.5",
                    "cell_phone_mAP.5", "person_mAP", "car_mAP", "motorcycle_mAP", "bus_mAP", "truck_mAP",
                    "baseball_bat_mAP", "knife_mAP", "cell_phone_mAP", "experiment_time"
                    ]
    exp1_perf = pd.DataFrame(columns=column_names)

    counter = 0
    for model in models:
        imgsize = 1280 if "6" in model else 640
        for precision in precisions:
            if precision == 'int8' and model == 'yolov5l6':
                break
            # for batch_size in BATCH_SIZES:
            start_experiment = datetime.now()
            row = [model, imgsize, precision]
            print(row)
            weights = f'openvino_models/{model}_{precision}_{imgsize}'
            print(weights)
            extra_metrics = []  # names, ap, ap50, ap_class, p, r, tp, fp
            glob_metrics, maps, t, extra_metrics = \
                run(data='data/coco.yaml', weights=weights, imgsz=imgsize, verbose=True)

            # process mAP results per class
            coco_class_index = {}
            for class_name in CLASSES:
                for k, v in extra_metrics[0].items():
                    if class_name == v:
                        coco_class_index[v] = k

            class_ap_index = {}
            for (k, v) in coco_class_index.items():
                class_ap_index[k] = np.where(extra_metrics[3] == v)[0][0]

            class_ap50 = {}
            class_ap = {}

            for (k, v) in class_ap_index.items():
                class_ap50[k] = extra_metrics[2][v]
                class_ap[k] = extra_metrics[1][v]
            row.append(glob_metrics[2])  # mAP50 all classes
            row.append(glob_metrics[3])  # mAP all classes
            row.append(np.mean(list(class_ap50.values())))
            row.append(np.mean(list(class_ap.values())))

            # append class specific mAPs
            for mAP50 in class_ap50.values():
                row.append(mAP50)
            for mAP in class_ap.values():
                row.append(mAP)

            # check if mAP equals mean of AP print(f'mAP50- old: {glob_metrics[2]} new: {np.mean(ap50)} mAP- old: {
            # glob_metrics[3]} new: {np.mean(ap)}')

            row.append((datetime.now() - start_experiment).seconds)  # duration of experiment
            exp1_perf.loc[counter] = row
            counter += 1
    print(exp1_perf)
    # store results
    filename = f'results/experiments/exp1/{datetime.now().strftime("%Y%m%d")}_exp1_perf'
    exp1_perf.round(3)
    print(exp1_perf)
    exp1_perf.to_pickle(filename + '.pkl')
    exp1_perf.to_csv(filename + '.csv')


if __name__ == "__main__":
    run_exp1_perf(['yolov5l6'], ['fp16', 'fp32'])
    run_exp1_speed(['yolov5l6'], ['fp16', 'fp32'])




