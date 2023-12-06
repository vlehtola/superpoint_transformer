import numpy as np
import os.path as osp

########################################################################
#                         Download information                         #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

FORM_URL = "https://docs.google.com"
ZIP_NAME = "ITC-KUL_Constructor_v1.1.zip"
UNZIP_NAME = "ITC-KUL_Constructor_v1.1"


########################################################################
#                              Data splits                             #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

VALIDATION_EPOCHS = [
    "2019",
    "WEEK17"]

EPOCHS = {
    "ITC_BUILDING": [
        "2019",
        "2021"],
       # "2022",
       # "2023"],
    "KUL_BUILDING": [
        "WEEK17",
        "WEEK18",
        "WEEK22"]
}


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

ITCKUL_NUM_CLASSES = 21

INV_OBJECT_LABEL = {
    0: "structural",
    1: "column",
    2: "beam",
    3: "floor",
    4: "wall",
    5: "stair",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "secondary",
    11: "ceiling",
    12: "door",
    13: "window",
    14: "railing",
    15: "MEP",
    16: "fire",
    17: "furniture",
    18: "18",
    19: "19",
    20: "20"
}

CLASS_NAMES = [INV_OBJECT_LABEL[i] for i in range(ITCKUL_NUM_CLASSES)] + ['ignored']

CLASS_COLORS = np.asarray([
    [233, 229, 107],  # 'structural'   ->  yellow
    [95, 156, 196],   # 'column'     ->  blue
    [179, 116, 81],   # ''      ->  brown
    [241, 149, 131],  # ''      ->  salmon
    [81, 163, 148],   # ''    ->  bluegreen
    [77, 174, 84],    # ''    ->  bright green
    [108, 135, 75],   # ''      ->  dark green
    [41, 49, 101],    # ''     ->  darkblue
    [79, 79, 76],     # ''     ->  dark grey
    [223, 52, 52],    # ''  ->  red
    [89, 47, 95],     # ''      ->  purple
    [81, 109, 114],   # ''     ->  grey
    [233, 233, 229],  # ''   ->  light grey
    [255, 255, 255,],  # white
    [255, 255, 255,],  # white
    [255, 255, 255,],  # white
    [255, 255, 255,],  # white
    [255, 255, 255,],  # white
    [255, 255, 255,],  # white
    [255, 255, 255,],  # white
    [0, 0, 0]])       # unlabelled  -> black

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

def object_name_to_label(object_class):
    """Convert from object name to int label. By default, if an unknown
    object nale
    """
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label
