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

root_data_dir = r'D:\Capstone\dataset'
# root_data_dir = r'/workspace/heart-monai/datasets'
data_dir = "/dataset-wholeheart/"
split_JSON = "dataset_" + modality + ".json"
datasets = root_data_dir + data_dir + split_JSON

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

train_ds = CacheDataset(
    data=datalist, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=num_workers
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

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
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
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):

        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        # print(logit_map)
        # print(y)
        # print(logit_map.shape)
        # print(y.shape)
        # return
        # logic_map is input and y is target
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
                global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), "best_metric_model_" + get_model_name(model) + "_" + modality + ".pth"
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


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
    max_iterations = 25000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=8)
    post_pred = AsDiscrete(argmax=True, to_onehot=8)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    print("model label: " + model_label + " | " + "model: " + get_model_name(model) + " | " + "modality: " + modality)

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    model.load_state_dict(torch.load("best_metric_model_" + get_model_name(model) + "_" + modality + ".pth"))
    print(
        f"train completed, best_metric: {dice_val_best:.4f}"
        f"at iteration: {global_step_best}"
    )

    a_file = open("epoch_loss_values_" + get_model_name(model) + "_" + modality + ".txt", "w+")
    epoch_loss_values = np.array(epoch_loss_values)
    print(epoch_loss_values)
    a_file.write(str(epoch_loss_values))
    a_file.close()

    b_file = open("metric_values_" + get_model_name(model) + "_" + modality + ".txt", "w+")
    metric_values = np.array(metric_values)
    print(metric_values)
    b_file.write(str(metric_values))
    b_file.close()
