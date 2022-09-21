import glob

import matplotlib.pyplot as plt
import torch
import numpy as np

from const import img_size
from models.unetr import UNETR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filepath_suffix = "D:\\Capstone\\heart-monai\\results\\monai@48\\attentionunet\\ct\\"
filepath_suffix = ""
filename_epoch = glob.glob(filepath_suffix + 'epoch_loss_values_*.txt')[0]
file = open(filename_epoch, "r")
epoch_loss_values = list(map(float, file.read()[1:-1].split()))
filename_metric = glob.glob(filepath_suffix + 'metric_values_*.txt')[0]
file = open(filename_metric, "r")
metric_values = list(map(float, file.read()[1:-1].split()))
model_suffix = "_".join(filename_metric.split('.')[0].split('_')[2:4])
eval_num = 500

# model = UNETR(
#     in_channels=1,
#     out_channels=8,
#     img_size=img_size,
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)

# filename_model = glob.glob('best_metric_model_*.pth')[0]
# # model.load_state_dict(torch.load("best_metric_model.pth"))
# model.load_state_dict(torch.load(filename_model, map_location=torch.device('cpu')),strict=False)
# model.eval()

def annot_minmax(x,y, minmax=1, ax=None):
    if minmax == 0:
        ymax = min(y)
        xmax = x[np.argmin(y)]
    else:
        ymax = max(y)
        xmax = x[np.argmax(y)]
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.50), **kw)

# monitoring
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
annot_minmax(x, y, 0)
ax=plt.gca()
# ax.set_ylim(top = 150)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
annot_minmax(x,y, 1)
ax=plt.gca()
#ax.set_ylim(top = 150)
plt.savefig(filepath_suffix + 'temp-model-monitoring_' + model_suffix + '.png')
plt.show()
