import numpy as np
import torch
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
)
from tqdm import tqdm

from const import img_size, train_transforms_ct, val_transforms_ct, train_transforms_mr, val_transforms_mr, model_unetr, \
    model_unet, model_attentionunet, num_workers
from models.attentionunet import AttentionUnet
from models.unetr import UNETR
from models.unet import UNet

# model_label = "unetr"
# model_label = "unet"
model_label = "attentionunet"

modality = "ct"
# modality = "mr"

if modality == "ct":
    val_transforms = val_transforms_ct
elif modality == "mr":
    val_transforms = val_transforms_mr

if model_label == "unetr":
    model = model_unetr
elif model_label == "unet":
    model = model_unet
elif model_label == "attentionunet":
    model = model_attentionunet

# root_data_dir = r'D:\Capstone\dataset'
root_data_dir = r'/workspace/heart-monai/datasets'
data_dir = "/dataset-wholeheart/"
split_JSON = "dataset_" + modality + ".json"
datasets = root_data_dir + data_dir + split_JSON

val_files = load_decathlon_datalist(datasets, True, "validation")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=num_workers
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

def validation():
    model.eval()
    dice_vals = list()
    metric_values_bg = list()
    metric_values_lv = list()
    metric_values_rv = list()
    metric_values_la = list()
    metric_values_ra = list()
    metric_values_myo = list()
    metric_values_ao = list()
    metric_values_pa = list()

    epoch_iterator_val = tqdm(
        val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    )

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, img_size, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            metric_batch = dice_metric_batch.aggregate()
            metric_bg = metric_batch[0].item()
            metric_values_bg.append(metric_bg)
            metric_lv = metric_batch[1].item()
            metric_values_lv.append(metric_lv)
            metric_rv = metric_batch[2].item()
            metric_values_rv.append(metric_rv)
            metric_la = metric_batch[3].item()
            metric_values_la.append(metric_la)
            metric_ra = metric_batch[4].item()
            metric_values_ra.append(metric_ra)
            metric_myo = metric_batch[5].item()
            metric_values_myo.append(metric_myo)
            metric_ao = metric_batch[6].item()
            metric_values_ao.append(metric_ao)
            metric_pa = metric_batch[7].item()
            metric_values_pa.append(metric_pa)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
        dice_metric_batch.reset()
    print(dice_vals)
    print(metric_values_bg)
    print(metric_values_lv)
    print(metric_values_rv)
    print(metric_values_la)
    print(metric_values_ra)
    print(metric_values_myo)
    print(metric_values_ao)
    print(metric_values_pa)
    mean_dice_val = np.mean(dice_vals)
    mean_metric_values_bg = np.mean(metric_values_bg)
    mean_metric_values_lv = np.mean(metric_values_lv)
    mean_metric_values_rv = np.mean(metric_values_rv)
    mean_metric_values_la = np.mean(metric_values_la)
    mean_metric_values_ra = np.mean(metric_values_ra)
    mean_metric_values_myo = np.mean(metric_values_myo)
    mean_metric_values_ao = np.mean(metric_values_ao)
    mean_metric_values_pa = np.mean(metric_values_pa)
    return mean_dice_val, mean_metric_values_bg, mean_metric_values_lv, mean_metric_values_rv, \
           mean_metric_values_la, mean_metric_values_ra, mean_metric_values_myo, mean_metric_values_ao, mean_metric_values_pa

def get_model_name(model):
    if isinstance(model, UNETR):
        model_name = "unetr"
    elif isinstance(model, UNet):
        model_name = "unet"
    elif isinstance(model, AttentionUnet):
        model_name = "attentionunet"
    else:
        model_name = "unknown"
    return model_name


if __name__ == '__main__':
    post_label = AsDiscrete(to_onehot=8)
    post_pred = AsDiscrete(argmax=True, to_onehot=8)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch",get_not_nans=False)
    global_step = 0
    global_step_best = 0
    metric_values = []

    print("model label: " + model_label + " | " + "model: " + get_model_name(model) + " | " + "modality: " + modality)

    model.load_state_dict(torch.load("best_metric_model_" + get_model_name(model) + "_" + modality + ".pth"), strict = False)

    dice_val, metric_values_bg, metric_values_lv, metric_values_rv, metric_values_la, metric_values_ra, metric_values_myo, \
    metric_values_ao, metric_values_pa = validation()

    metric_values.append(dice_val)
    metric_values.append(metric_values_bg)
    metric_values.append(metric_values_lv)
    metric_values.append(metric_values_rv)
    metric_values.append(metric_values_la)
    metric_values.append(metric_values_ra)
    metric_values.append(metric_values_myo)
    metric_values.append(metric_values_ao)
    metric_values.append(metric_values_pa)

    print(
        f"train completed, overall dice val: {dice_val:.4f} & dice background: {metric_values_bg:.4f} & dice LV: {metric_values_lv:.4f} & dice RV: {metric_values_rv:.4f} "
        f"& dice LA: {metric_values_la:.4f} & dice RA: {metric_values_ra:.4f} & dice Myo: {metric_values_myo:.4f} & dice AO: {metric_values_ao:.4f} & dice PA: {metric_values_pa:.4f}"
    )

    results_file = open("val_model_metrics_" + get_model_name(model) + "_" + modality + ".txt", "w+")
    print(metric_values)
    results_file.write(str(metric_values))
    results_file.close()
