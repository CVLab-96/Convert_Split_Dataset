import os
import json
import shutil
from PIL import Image

def yolo_to_coco(yolo_root, coco_root, splits, class_names):
    """
    将YOLO格式数据集转换为COCO格式

    参数:
        yolo_root: YOLO格式数据集根目录
        coco_root: 输出COCO格式数据集根目录
        splits: 需要转换的数据集划分列表 (['train', 'val', 'test'])
        class_names: 类别名称列表 (按YOLO类别索引顺序)
    """
    # 创建COCO目录结构
    os.makedirs(os.path.join(coco_root, 'annotations'), exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(coco_root, 'images', f'{split}2014'), exist_ok=True)

    # 处理每个数据集划分
    for split in splits:
        coco_data = {
            "info": {"description": "COCO Dataset converted from YOLO"},
            "licenses": [{"name": "Unknown"}],
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "none"} 
                          for i, name in enumerate(class_names)]
        }

        image_id = 1
        annotation_id = 1

        # 遍历YOLO图片目录
        yolo_img_dir = os.path.join(yolo_root, split, 'images')
        yolo_label_dir = os.path.join(yolo_root, split, 'labels')

        for img_name in os.listdir(yolo_img_dir):
            # 处理图片文件
            img_path = os.path.join(yolo_img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(yolo_label_dir, f"{base_name}.txt")

            # 获取图片尺寸
            with Image.open(img_path) as img:
                width, height = img.size

            # 添加图片信息
            coco_image = {
                "id": image_id,
                "file_name": img_name,  # 修改为只使用文件名
                "width": width,
                "height": height
            }
            coco_data["images"].append(coco_image)

            # 处理标签文件
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        # 解析YOLO格式
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # 转换为COCO格式的绝对坐标
                        abs_x = (x_center - w/2) * width
                        abs_y = (y_center - h/2) * height
                        abs_w = w * width
                        abs_h = h * height

                        # 转换为整数
                        bbox = [round(abs_x), round(abs_y), 
                               round(abs_w), round(abs_h)]

                        # 添加标注信息
                        coco_ann = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": bbox,
                            "area": abs_w * abs_h,
                            "iscrowd": 0
                        }
                        coco_data["annotations"].append(coco_ann)
                        annotation_id += 1

            # 复制图片到COCO目录
            dst_path = os.path.join(coco_root, 'images', f'{split}2014', img_name)
            shutil.copy(img_path, dst_path)

            image_id += 1

        # 保存JSON文件
        output_path = os.path.join(coco_root, 'annotations', 
                                 f'instances_{split}2014.json')
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

# 使用示例
if __name__ == "__main__":
    # 配置参数
    YOLO_ROOT = "./TACO_YOLO"
    COCO_ROOT = "./yolo2coco"
    SPLITS = ["train", "val", "test"]
    CLASS_NAMES = ['Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton', 'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper', 'Plastic bag - wrapper', 'Plastic container', 'Pop tab', 'Straw', 'Styrofoam piece', 'Unlabeled litter']  # 替换为实际类别名称

    # 执行转换
    yolo_to_coco(YOLO_ROOT, COCO_ROOT, SPLITS, CLASS_NAMES)    