import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm


def coco_to_yolo(coco_root, yolo_output):
    # 创建YOLO目录结构
    splits = {"train2014": "train", "val2014": "val"}
    for s in splits.values():
        os.makedirs(os.path.join(yolo_output, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_output, s, "labels"), exist_ok=True)

    # 保存类别文件
    coco = COCO(os.path.join(coco_root, "annotations/instances_train2014.json"))
    class_list = [coco.cats[cat_id]['name'] for cat_id in sorted(coco.cats.keys())]
    with open(os.path.join(yolo_output, "classes.txt"), 'w') as f:
        f.write("\n".join(class_list))

    # 处理每个split
    for coco_split, yolo_split in splits.items():
        coco = COCO(os.path.join(coco_root, f"annotations/instances_{coco_split}.json"))

        for img_id in tqdm(coco.imgs, desc=f"Processing {coco_split}"):
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # 生成YOLO标签文件
            txt_path = os.path.join(yolo_output, yolo_split, "labels",
                                    f"{os.path.splitext(img_info['file_name'])[0]}.txt")
            with open(txt_path, 'w') as f:
                for ann in anns:
                    class_id = sorted(coco.cats.keys()).index(ann['category_id'])
                    x, y, w, h = ann['bbox']
                    x_center = (x + w / 2) / img_info['width']
                    y_center = (y + h / 2) / img_info['height']
                    bw = w / img_info['width']
                    bh = h / img_info['height']
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")


if __name__ == "__main__":
    coco_to_yolo(
        coco_root="./datasets/coco2014",
        yolo_output="./coco2yolo"
    )
