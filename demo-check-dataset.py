import os

import matplotlib.pyplot as plt
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)

from const import train_transforms_ct, val_transforms_ct, train_transforms_mr, val_transforms_mr, num_workers

# modality = "ct"
modality = "mr"

if modality == "ct":
    train_transforms = train_transforms_ct
    val_transforms = val_transforms_ct
elif modality == "mr":
    train_transforms = train_transforms_mr
    val_transforms = val_transforms_mr

train_transforms = train_transforms
val_transforms = val_transforms

root_dir = r'D:\Capstone\dataset'
data_dir = "/dataset-wholeheart/"
split_JSON = "dataset_"+modality+".json"
datasets = root_dir + data_dir + split_JSON

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

if modality == "ct":
    slice_map = {
        # # val
        "ct_train_1017_image.nii.gz": 50,
        "ct_train_1018_image.nii.gz": 30,
        "ct_train_1019_image.nii.gz": 50,
        "ct_train_1020_image.nii.gz": 50,

        # # train
        # # "ct_train_1001_image.nii.gz": 50,
        # # "ct_train_1002_image.nii.gz": 50,
        # # "ct_train_1003_image.nii.gz": 50,
        # # "ct_train_1004_image.nii.gz": 50,
    }
elif modality == "mr":
    slice_map = {

        # val
        "mr_train_1017_image.nii.gz": 50,
        "mr_train_1018_image.nii.gz": 30,
        "mr_train_1019_image.nii.gz": 50,
        "mr_train_1020_image.nii.gz": 50,

        # train
        # "mr_train_1001_image.nii.gz": 50,
        # "mr_train_1002_image.nii.gz": 50,
        # "mr_train_1003_image.nii.gz": 50,
        # "mr_train_1004_image.nii.gz": 50,
    }

for i in range(len(slice_map)):
    if(i == 0):
        case_num = i
        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        img_shape = img.shape
        label_shape = label.shape
        print(f"image shape: {img_shape}, label shape: {label_shape}")
        for i in [-20,5,5,5,5,5,5,5,5]:
        #for i in [-20,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]:
            slice_map[img_name] = slice_map[img_name]+i
            plt.figure("image", (18, 6))
            plt.subplot(1, 2, 1)
            plt.title("image: "+ img_name + '-' + "slice: " +  str(slice_map[img_name]))
            plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("label: " + img_name + '-' + "slice: " +  str(slice_map[img_name]))
            plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
            plt.show()