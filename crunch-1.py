#import crunch
from train1 import *
from infer1 import *
#crunch = crunch.load_notebook()
# /!\ Don't forget to import pre-trained Resnet50 "pytorch_model.bin" in model_directory_path="./resources" from https://huggingface.co/timm/resnet50.tv_in1k/tree/main
#train1(data_directory_path='./data.2.large', model_directory_path="./resources")
prediction = infer1(
    data_file_path="./data/UC9_I.zarr",
    model_directory_path="./resources"
)
print(prediction.head())
if 0:
    crunch.test(no_determinism_check=True)