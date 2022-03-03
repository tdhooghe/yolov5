#%%
import torch
import os

print(os.getcwd())
#%%
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')  # or yolov5m, yolov5l, yolov5x, custom


# images
img = f'{os.getcwd()}/data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
#img = '/mnt/nas/private/photos/vakanties/2013 - Cavalaire sur Mer (St. Tropez)/DSC00003.JPG'
# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()