import os
from pathlib import Path

from val import run
from datetime import datetime
import numpy as np
import pandas as pd

MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l",
          # "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6"
          ]
# %%
#CLASSES = ['person', 'car', 'motorcycle', 'bus', 'truck', 'baseball bat', 'knife', 'cell phone']


# train models only with above classes, remove the others; the previous experiment is about general selection so then
# the other classes do no matter

def model_validation(models, classes, validation_set):
    # glob_metrics: mp, mr, map50, map
    # t: prep., inference, NMS
    # person: 0, car: 2, motorcycle: 3, bus: 5, truck: 7, baseball bat: 38, knife: 48, cell phone: 76
    imgsize = 1280
    column_names = ["model", "img_size", "mAP05_all", "mAP_all", "mAP05_classes",
                    "mAP_classes", "experiment_time"]
    column_names.extend([f'{x}_mAP50' for x in classes])
    column_names.extend([f'{x}_mAP' for x in classes])
    print(column_names)
    validation_results = pd.DataFrame(columns=column_names)

    for model in models:
        counter = 0

        start_experiment = datetime.now()
        row = [model, imgsize]
        print(row)
        glob_metrics, maps, t, names, ap, ap50, ap_class, p, r, tp, fp = \
            run(data=validation_set, weights=model, imgsz=imgsize, verbose=True)

        # process mAP results per class
        coco_class_index = {}
        for class_name in classes:
            for k, v in names.items():
                if class_name == v:
                    coco_class_index[v] = k

        class_ap_index = {}
        for (k, v) in coco_class_index.items():
            class_ap_index[k] = np.where(ap_class == v)[0][0]

        class_ap50 = {}
        class_ap = {}

        for (k, v) in class_ap_index.items():
            class_ap50[k] = ap50[v]
            class_ap[k] = ap[v]

        row.append(glob_metrics[2])  # mAP50 all classes
        row.append(glob_metrics[3])  # mAP all classes
        row.append(np.mean(list(class_ap50.values())))
        row.append(np.mean(list(class_ap.values())))

        # check if mAP equals mean of AP print(f'mAP50- old: {glob_metrics[2]} new: {np.mean(ap50)} mAP- old: {
        # glob_metrics[3]} new: {np.mean(ap)}')

        row.append((datetime.now() - start_experiment).seconds)  # duration of experiment

        # append class specific mAPs
        for mAP50 in class_ap50.values():
            row.append(mAP50)
        for mAP in class_ap.values():
            row.append(mAP)

        validation_results.loc[counter] = row
        counter += 1
    validation_results.round(3)

    # store results
    filename = Path(f'results/dataset/{datetime.now().strftime("%y%m%d-%H-%M")}_' \
               f'{os.path.splitext(os.path.split(validation_set)[-1])[0]}_' \
               f'{os.path.splitext(os.path.split(model)[-1])[0]}')
    filename.parent.mkdir(parents=True, exist_ok=True)

    # validation_results.to_pickle(filename + '.pkl')
    validation_results.to_csv(str(filename) + '.csv')


if __name__ == "__main__":
    model_validation(['yolov5n6.pt'], ['person', 'car', 'motorcycle', 'bus', 'truck', 'knife', 'cell phone'], 'data/coco128.yaml')

