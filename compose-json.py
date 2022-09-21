import glob
import json

data_dir = r'D:\Capstone\dataset'

ct_train_dst = data_dir + r'\dataset-wholeheart\imagesTr'
ct_label_dst = data_dir + r'\dataset-wholeheart\labelsTr'
ct_test_dst = data_dir + r'\dataset-wholeheart\imagesTs'

mr_train_dst = data_dir + r'\dataset-wholeheart\mr_imagesTr'
mr_label_dst = data_dir + r'\dataset-wholeheart\mr_labelsTr'
mr_test_dst = data_dir + r'\dataset-wholeheart\mr_imagesTs'

validation_split = 0.2

def json_conversion_info(image, label):
    info = {
        "image": image,
        "label": label,
    }
    return info

def create_json_ct():
    json_conversion_output = {}

    json_conversion_output["description"] = "dataset-whole-heart-segmentation jk"
    json_conversion_output["labels"] = {
        #  LV, RV, LA, RA, Myo, AO, PA are labeled 500, 600, 420, 550, 205, 820, 850
        "0": "background",
        "1": "LV",
        "2": "RV",
        "3": "LA",
        "4": "RA",
        "5": "Myo",
        "6": "AO",
        "7": "PA",
    }
    json_conversion_output["licence"] = "yt"

    json_conversion_output["modality"] = {
        "0": "CT"
    }

    json_conversion_output["name"] = "whs"
    json_conversion_output["numTest"] = ""
    json_conversion_output["numTraining"] = ""
    json_conversion_output["reference"] = "Vanderbilt University"
    json_conversion_output["release"] = "1.0 06/08/2015"
    json_conversion_output["tensorImageSize"] = "3D"
    json_conversion_output["test"] = []
    json_conversion_output["training"] = []
    json_conversion_output["validation"] = []

    # manual copy into folders
    train_image_fps = list(sorted(glob.glob(ct_train_dst + '/'+ '*.gz')))
    train_image_fps_id = [ ('imagesTr'+ '/' + i.split('\\')[-1]) for i in train_image_fps]
    train_label_fps = list(sorted(glob.glob(ct_label_dst + '/' + '*.gz')))
    train_label_fps_id = [('labelsTr' + '/' + i.split('\\')[-1]) for i in train_label_fps]
    test_image_fps = list(sorted(glob.glob(ct_test_dst + '/' + '*.gz')))
    test_image_fps_id = [('imagesTs' + '/' + i.split('\\')[-1]) for i in test_image_fps]

    json_conversion_output["numTest"] = len(test_image_fps_id)
    json_conversion_output["numTraining"] = len(train_image_fps_id) + len(train_label_fps_id) + len(test_image_fps_id)

    for item in test_image_fps_id:
        json_conversion_output["test"].append(item)

    split_index = int((1 - validation_split) * len(train_image_fps_id))
    image_fps_train = train_image_fps_id[:split_index]
    image_fps_val = train_image_fps_id[split_index:]

    image_fps_train_labels = train_label_fps_id[:split_index]
    image_fps_val_labels = train_label_fps_id[split_index:]

    for i in range(len(image_fps_train)):
        json_conversion_output["training"].append(json_conversion_info(image_fps_train[i],image_fps_train_labels[i]))
    for i in range((len(image_fps_val))):
        json_conversion_output["validation"].append(json_conversion_info(image_fps_val[i], image_fps_val_labels[i]))

    output_file_name = 'dataset_ct.json'

    with open(output_file_name, "w") as f:
        json.dump(json_conversion_output, f)

    return json_conversion_output

def create_json_mr():
    json_conversion_output = {}

    json_conversion_output["description"] = "dataset-whole-heart-segmentation jk"
    json_conversion_output["labels"] = {
        #  LV, RV, LA, RA, Myo, AO, PA are labeled 500, 600, 420, 550, 205, 820, 850
        "0": "background",
        "1": "LV",
        "2": "RV",
        "3": "LA",
        "4": "RA",
        "5": "Myo",
        "6": "AO",
        "7": "PA",
    }
    json_conversion_output["licence"] = "yt"
    json_conversion_output["modality"] = {
        "0": "MR"
    }
    json_conversion_output["name"] = "whs"
    json_conversion_output["numTest"] = ""
    json_conversion_output["numTraining"] = ""
    json_conversion_output["reference"] = "Vanderbilt University"
    json_conversion_output["release"] = "1.0 06/08/2015"
    json_conversion_output["tensorImageSize"] = "3D"
    json_conversion_output["test"] = []
    json_conversion_output["training"] = []
    json_conversion_output["validation"] = []

    # manual copy into folders
    train_image_fps = list(sorted(glob.glob(mr_train_dst + '/'+ '*.gz')))
    train_image_fps_id = [ ('mr_imagesTr'+ '/' + i.split('\\')[-1]) for i in train_image_fps]
    train_label_fps = list(sorted(glob.glob(mr_label_dst + '/' + '*.gz')))
    train_label_fps_id = [('mr_labelsTr' + '/' + i.split('\\')[-1]) for i in train_label_fps]
    test_image_fps = list(sorted(glob.glob(mr_test_dst + '/' + '*.gz')))
    test_image_fps_id = [('mr_imagesTs' + '/' + i.split('\\')[-1]) for i in test_image_fps]

    json_conversion_output["numTest"] = len(test_image_fps_id)
    json_conversion_output["numTraining"] = len(train_image_fps_id) + len(train_label_fps_id) + len(test_image_fps_id)

    for item in test_image_fps_id:
        json_conversion_output["test"].append(item)

    split_index = int((1 - validation_split) * len(train_image_fps_id))
    image_fps_train = train_image_fps_id[:split_index]
    image_fps_val = train_image_fps_id[split_index:]

    image_fps_train_labels = train_label_fps_id[:split_index]
    image_fps_val_labels = train_label_fps_id[split_index:]

    for i in range(len(image_fps_train)):
        json_conversion_output["training"].append(json_conversion_info(image_fps_train[i],image_fps_train_labels[i]))
    for i in range((len(image_fps_val))):
        json_conversion_output["validation"].append(json_conversion_info(image_fps_val[i], image_fps_val_labels[i]))

    output_file_name = 'dataset_mr.json'

    with open(output_file_name, "w") as f:
        json.dump(json_conversion_output, f)

    return json_conversion_output

create_json_ct()
create_json_mr()

