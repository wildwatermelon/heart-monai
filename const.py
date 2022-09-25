import torch
from monai.networks.layers import Norm
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    Spacingd,
    RandRotate90d,
    ToTensord,
    MapLabelValued, RandShiftIntensityd, ScaleIntensityRanged, NormalizeIntensityd, RandScaleIntensityd,
)

from models.attentionunet import AttentionUnet
from models.unet import UNet
from models.unetr import UNETR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 8

# original
# img_size = (48, 48, 48)
# ct_pixdim = (1.5, 1.5, 2.0)
# mr_pixdim = (1.5, 1.5, 2.0)

# goldilocks + intensity scaling
# img_size = (48, 48, 48)
# ct_pixdim = (2.0, 2.0, 2.0)
# mr_pixdim = (2.5, 2.5, 4.0)

# local
# img_size = (48, 48, 48)
# ct_pixdim = (2.0, 2.0, 2.0)
# mr_pixdim = (2.0, 2.0, 2.0)

img_size = (96, 96, 96)
ct_pixdim = (1.0, 1.0, 1.0)
mr_pixdim = (1.0, 1.0, 1.0)

# scale intensity baselines
# ct_scale_intensity_a_min = -175.0
# ct_scale_intensity_a_max = 250.0

# mr_scale_intensity_a_min = -175.0
# mr_scale_intensity_a_max = 250.0

model_unetr = UNETR(
    in_channels=1,
    out_channels=8,
    img_size=img_size,
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

model_unet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=8,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model_attentionunet = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=8,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

train_transforms_ct = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=ct_pixdim,
            mode=("bilinear", "nearest"),
        ),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=ct_scale_intensity_a_min,
        #     a_max=ct_scale_intensity_a_max,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        MapLabelValued(keys=["image", "label"],
                       orig_labels=[0.0, 500.0, 600.0, 420.0, 550.0, 205.00, 820.00, 850.0],
                       target_labels=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"])
    ]
)

val_transforms_ct = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=ct_pixdim,
            mode=("bilinear", "nearest"),
        ),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=ct_scale_intensity_a_min,
        #     a_max=ct_scale_intensity_a_max,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        MapLabelValued(keys=["image", "label"],
                       orig_labels=[0.0, 500.0, 600.0, 420.0, 550.0, 205.00, 820.00, 850.0],
                       target_labels=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ToTensord(keys=["image", "label"]),
    ]
)

train_transforms_mr = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=mr_pixdim,
            mode=("bilinear", "nearest"),
        ),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=mr_scale_intensity_a_min,
        #     a_max=mr_scale_intensity_a_max,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        MapLabelValued(keys=["image", "label"],
                       orig_labels=[0.0, 500.0, 600.0, 420.0, 421.0, 550.0, 205.00, 820.00, 850.0],
                       target_labels=[0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.10,
            prob=0.50
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"])
    ]
)

val_transforms_mr = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=mr_pixdim,
            mode=("bilinear", "nearest"),
        ),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=mr_scale_intensity_a_min,
        #     a_max=mr_scale_intensity_a_max,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        MapLabelValued(keys=["image", "label"],
                       orig_labels=[0.0, 500.0, 600.0, 420.0, 421.0, 550.0, 205.00, 820.00, 850.0],
                       target_labels=[0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.10,
            prob=0.50
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)
