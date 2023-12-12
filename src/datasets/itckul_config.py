import numpy as np
import os.path as osp

########################################################################
#                         Download information                         #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

FORM_URL = "https://docs.google.com"
ZIP_NAME = "ITC-KUL constructor.zip"
UNZIP_NAME = "ITC-KUL constructor"


########################################################################
#                              Data splits                             #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

VALIDATION_EPOCHS = [
    "WEEK17R"]

TRAIN_EPOCHS =  ["WEEK18"]

TEST_EPOCHS =  ["WEEK17"]

# connects the epoch label to the building dir
EPOCHS = {
        "2019": "ITC_BUILDING",
        "2021": "ITC_BUILDING",
        "2022": "ITC_BUILDING",
        "2023": "ITC_BUILDING",
        "WEEK17": "KUL_BUILDING",
        "WEEK17R": "KUL_BUILDING",
        "WEEK17RR": "KUL_BUILDING",
        "WEEK18": "KUL_BUILDING",
        "WEEK22": "KUL_BUILDING"
}


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

ITCKUL_NUM_CLASSES = 100

INV_OBJECT_LABEL = {
0 : "Structural",
1 : "Columns",
2 : "Beams",
3 : "Floors",
4 : "Walls",
5 : "Stairs",
6 : "Roofs",
7 : "CurtainWalls",
11 : "Ceilings",
12 : "Doors",
13 : "Windows",
14 : "Railings",
17 : "Furniture",
20 : "Building_materials",
21 : "Formwork",
22 : "Rebar",
23 : "Wood",
24 : "Brick",
25 : "Concrete",
26 : "Steel",
27 : "Aggregates",
30 : "Construction_equipment",
31 : "Crane",
32 : "Mixer",
33 : "Truck",
34 : "Excavator",
35 : "Telehandler",
36 : "Generator",
37 : "Scaffold",
40 : "Workers",
50 : "Site_layout",
51 : "Container",
52 : "Toilettes",
53 : "Fence",
54 : "Guardrails",
55 : "Traffic_signs",
56 : "Trash_bin",
57 : "Safety_mesh",
60 : "Outdoor",
61 : "Landscaping",
62 : "Vegetation",
63 : "Ground",
64 : "Water"    
}

CLASS_NAMES = [INV_OBJECT_LABEL.get(i, i) for i in range(ITCKUL_NUM_CLASSES)]+ ['ignored']

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
