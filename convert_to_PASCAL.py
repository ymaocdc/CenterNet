import os, glob, json
from tqdm import tqdm
import datetime
import numpy as np


mode = 'train'

root_folder = '/xmotors_ai_shared/datasets/incubator/user/yus/dataset/pku'
image_folder = os.path.join(root_folder, '{}_images'.format(mode))
label_folder = os.path.join(root_folder, '{}_labels'.format(mode))

__CLASS__ = ['__background__', '2x', '3x', 'SUV']
# __CLASS__ = ['bg', 'car']
annotations = {}

# coco annotations info.
annotations["info"] = {
    "description": "customer dataset format convert to COCO format",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2019,
    "contributor": "andy.wei",
    "date_created": "2019/01/24"
}
# coco annotations licenses.
annotations["licenses"] = [{
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    "id": 1,
    "name": "Apache License 2.0"
}]

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

im_files = glob.glob(os.path.join(image_folder, '*.jpg'))
im_files.sort()
label_files = glob.glob(os.path.join(label_folder, '*.json'))
label_files.sort()
assert len(im_files) == len(label_files)

annotations["images"] = []
annotations["annotations"] = []
for image_id, im_file in enumerate(im_files):
    width = 3384
    height = 2710
    file_name = im_file.split('/')[-1]
    annotations["images"].append(
        {
            "license": 1,
            "file_name": file_name,
            "coco_url": "",
            "height": height,
            "width": width,
            "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "flickr_url": "",
            "id": image_id
        }
    )

    label_file = label_files[image_id]
    with open(label_file, 'r') as fin:
        gt = json.load(fin)
    for obj_idx, obj in enumerate(gt['objects']):
        cls = np.argwhere(annotations["categories"] == obj['car_type'])
        xmin, ymin, w, h = obj['2D_bbox_xyxy'][0], obj['2D_bbox_xyxy'][1], obj['2D_bbox_xywh'][2], obj['2D_bbox_xywh'][3]
        # coco annotations annotations.
        annotations["annotations"].append(
            {
                "id": obj_idx,
                "image_id": image_id,
                "category_id": cls + 1,
                "segmentation": [[]],
                "area": w * h,
                "iscrowd": 0,
                "bbox": obj['2D_bbox_xyxy'],
                'projected_3D_center': obj['projected_3D_center'],
                'local_yaw': obj['local_yaw'],
                'theta': obj['theta'],
                'pitch': obj['pitch'],
                'roll': obj['roll'],
                'global_yaw': obj['global_yaw'],
                '3D_dimension': obj['3D_dimension'],
                'BPE': [obj['BPE_left'][0], obj['BPE_right'][1]],
                'FPE': [obj['FPE_left'][0], obj['FPE_right'][1]]
            }
        )

json_path = os.path.join(root_folder, '{}_coco_format'.format(mode)+".json")
with open(json_path, "w") as f:
    json.dump(annotations, f)
