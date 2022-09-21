# https://github.com/Project-MONAI/tutorials/blob/main/modules/dynunet_pipeline/calculate_task_params.py
# https://github.com/Project-MONAI/MONAI/discussions/2583

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from monai.data import (
    Dataset,
    DatasetSummary,
    load_decathlon_datalist,
    load_decathlon_properties,
)
from monai.transforms import LoadImaged

from const import num_workers

# modality = "ct"
modality = "mr"

def get_task_params(args):
    """
    This function is used to achieve the spacings of decathlon dataset.
    In addition, for CT images (task 03, 06, 07, 08, 09 and 10), this function
    also prints the mean and std values (used for normalization), and the min (0.5 percentile)
    and max(99.5 percentile) values (used for clip).
    """
    datalist_path = args.datalist_path

    # get all training data
    datalist = load_decathlon_datalist(
        datalist_path, True, "training"
    )

    # get modality info.
    properties = load_decathlon_properties(
        datalist_path, "modality"
    )

    dataset = Dataset(
        data=datalist,
        transform=LoadImaged(keys=["image", "label"]),
    )

    calculator = DatasetSummary(dataset, num_workers=num_workers)
    target_spacing = calculator.get_target_spacing()
    print("spacing: ", target_spacing)
    if properties["modality"]["0"] == "CT" or properties["modality"]["0"] == "MR":
        print("CT input, calculate statistics:")
        calculator.calculate_statistics()
        print("mean: ", calculator.data_mean, " std: ", calculator.data_std)
        calculator.calculate_percentiles(
            sampling_flag=True, interval=10, min_percentile=0.01, max_percentile=99.99
        )
        print(
            "min: ",
            calculator.data_min_percentile,
            " max: ",
            calculator.data_max_percentile,
        )
    else:
        print("non CT input, skip calculating.")

if __name__ == "__main__":
    root_data_dir = r'D:\Capstone\dataset'
    # root_data_dir = r'/workspace/heart-monai/datasets'
    data_dir = "/dataset-wholeheart/"
    split_JSON = "dataset_"+modality+".json"
    datasets = root_data_dir + data_dir + split_JSON
    # datasets = "D:\Capstone\dataset\Task09_Spleen\dataset.json"
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--datalist_path",
        type=str,
        default=datasets,
    )
    args = parser.parse_args()
    get_task_params(args)


# MR: 99.99 0.01
# spacing:  (0.9722222, 0.9722222, 1.05)
# CT input, calculate statistics:
# mean:  725.6836547851562  std:  313.0063781738281
# min:  3.0  max:  1941.252199999988

# CT: 99.99 0.01
# spacing:  (0.4384765, 0.4384765, 0.625)
# CT input, calculate statistics:
# mean:  261.0399475097656  std:  160.904296875
# min:  -224.0  max:  739.0




