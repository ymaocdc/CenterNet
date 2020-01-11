import os, glob, json
from tqdm import tqdm
import datetime
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Helper class to help serialize numpy ndarray"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


apollo = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/apollo/data/train_coco_format.json'
kaggle = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku/data/annotations/train_coco_format_correct_yaw.json'

with open(apollo, 'r') as a:
    gta = json.load(a)
with open(kaggle, 'r') as b:
    gtk = json.load(b)

annotations = {}

# coco annotations info.
annotations["info"] = {
    "description": "customer dataset format convert to COCO format",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2019,
    "contributor": "YMAO",
    "date_created": "2019/01/24"
}
# coco annotations licenses.
annotations["licenses"] = [{
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    "id": 1,
    "name": "Apache License 2.0"
}]
__CLASS__ = ['__background__', '2x', '3x', 'SUV']
# coco annotations categories.
annotations["categories"] = []
for cls, clsname in enumerate(__CLASS__):
    if clsname == '__background__':
        continue
    annotations["categories"].append(
        {
            "supercategory": "object",
            "id": cls,
            "name": clsname
        }
    )

annotations["images"] = gta['images'] + gtk['images']
annotations["annotations"] = gta['annotations'] + gtk['annotations']




json_path = os.path.join('/xmotors_ai_shared/datasets/incubator/user/yus/dataset/apollo/data/', 'combine_gt_train.json')
with open(json_path, "w") as f:
    json.dump(annotations, f,cls=NumpyEncoder)