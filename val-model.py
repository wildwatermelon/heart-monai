import os

import matplotlib.pyplot as plt
import torch
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
from monai.inferers import sliding_window_inference

from const import img_size, train_transforms_ct, val_transforms_ct, train_transforms_mr, val_transforms_mr, model_unetr, \
    model_unet, model_attentionunet, num_workers
from train import get_model_name

model_label = "unetr"
# model_label = "unet"
# model_label = "attentionunet"

modality = "ct"
# modality = "mr"

if modality == "ct":
    train_transforms = train_transforms_ct
    val_transforms = val_transforms_ct
elif modality == "mr":
    train_transforms = train_transforms_mr
    val_transforms = val_transforms_mr

if model_label == "unetr":
    model = model_unetr
elif model_label == "unet":
    model = model_unet
elif model_label == "attentionunet":
    model = model_attentionunet

train_transforms = train_transforms
val_transforms = val_transforms

root_data_dir = r'D:\Capstone\dataset'
# root_data_dir = r'/workspace/heart-monai/datasets'
data_dir = "/dataset-wholeheart/"
split_JSON = "dataset_"+modality+".json"
datasets = root_data_dir + data_dir + split_JSON

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

train_ds = CacheDataset(
    data=datalist,transform=train_transforms,cache_num=24,cache_rate=1.0,num_workers=num_workers
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True
)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=num_workers
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slice_map = {
    # val
    "ct_train_1017_image.nii.gz": 50,
    "ct_train_1018_image.nii.gz": 50,
    "ct_train_1019_image.nii.gz": 50,
    "ct_train_1020_image.nii.gz": 50,

    # train
    # "ct_train_1001_image.nii.gz": 50,
    # "ct_train_1002_image.nii.gz": 50,
    # "ct_train_1003_image.nii.gz": 50,
    # "ct_train_1004_image.nii.gz": 50,

    # val 120
    # "mr_train_1017_image.nii.gz": 50,
    # "mr_train_1018_image.nii.gz": 50,
    # "mr_train_1019_image.nii.gz": 50,
    # "mr_train_1020_image.nii.gz": 50,

    # train 120
    # "mr_train_1001_image.nii.gz": 50,
    # "mr_train_1002_image.nii.gz": 50,
    # "mr_train_1003_image.nii.gz": 50,
    # "mr_train_1004_image.nii.gz": 50,
}


case_num = 1
#model.load_state_dict(torch.load("best_metric_model_" +get_model_name(model)+ "_" +modality+ ".pth"))
model.load_state_dict(torch.load("best_metric_model_" +get_model_name(model)+ "_" +modality+ ".pth", map_location=torch.device('cpu')),strict=False)
print("using: "+"best_metric_model_" +get_model_name(model)+ "_" +modality+ ".pth")
model.eval()
with torch.no_grad():
    img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]

    # val_inputs = torch.unsqueeze(img, 1).cuda()
    # val_labels = torch.unsqueeze(label, 1).cuda()
    val_inputs = torch.unsqueeze(img, 1)
    val_labels = torch.unsqueeze(label, 1)

    val_outputs = sliding_window_inference(
        val_inputs, img_size, 4, model, overlap=0.8
    )
    for i in [-20, 5, 5, 5, 5, 5, 5, 5, 5]:
        slice_map[img_name] = slice_map[img_name] + i
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image: "+ img_name + '-' + "slice: " +  str(slice_map[img_name]))
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label: " + img_name + '-' + "slice: " +  str(slice_map[img_name]))
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
        plt.subplot(1, 3, 3)
        plt.title("output: " + img_name + '-' + "slice: " +  str(slice_map[img_name]))
        plt.imshow(
            torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
        )
        plt.savefig('temp-model-validation-'+img_name+'-'+get_model_name(model)+ "-" +modality+ "-" +str(slice_map[img_name])+'.png')
        plt.show()