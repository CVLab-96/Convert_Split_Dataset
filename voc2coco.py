import os
import json
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm


def voc_to_coco(voc_root, output_dir="COCODataset"):
    # 创建COCO目录结构
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    os.makedirs(f"{output_dir}/images/train2014", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val2014", exist_ok=True)

    # 收集所有类别
    categories = set()
    for xml_file in os.listdir(f"{voc_root}/Annotations"):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(f"{voc_root}/Annotations/{xml_file}")
        root = tree.getroot()
        for obj in root.findall("object"):
            categories.add(obj.find("name").text)

    # 创建类别映射
    cat_id_map = {name: i+1 for i, name in enumerate(sorted(categories))}
    with open(f"{output_dir}/classes.txt", "w") as f:
        f.write("\n".join(sorted(categories)))

    # 处理每个划分集
    for split in ["train", "val", "test"]:
        split_file = f"{voc_root}/ImageSets/Main/{split}.txt"
        if not os.path.exists(split_file):
            continue

        # 初始化COCO数据结构
        coco_data = {
            "info": {"description": "COCO Dataset", "year": 2023},
            "licenses": [{"id": 1}],
            "categories": [{"id": v, "name": k} for k, v in cat_id_map.items()],
            "images": [],
            "annotations": []
        }

        # 读取划分集
        with open(split_file) as f:
            image_names = [line.strip() for line in f.readlines()]

        ann_id = 0
        for img_id, base_name in tqdm(enumerate(image_names), desc=f"Processing {split} set"):
            # 处理图片
            src_img = f"{voc_root}/JPEGImages/{base_name}.jpg"
            if not os.path.exists(src_img):
                continue

            # 复制图片
            dst_folder = "train2014" if split == "train" else "val2014"
            dst_img = f"{output_dir}/images/{dst_folder}/{base_name}.jpg"
            shutil.copy(src_img, dst_img)

            # 获取图片尺寸
            with Image.open(src_img) as img:
                width, height = img.size

            # 添加图片信息（修改点1：file_name不含路径）
            coco_data["images"].append({
                "id": img_id,
                "file_name": f"{base_name}.jpg",  # 仅保留文件名
                "width": width,
                "height": height,
                "license": 1
            })

            # 处理标注
            xml_path = f"{voc_root}/Annotations/{base_name}.xml"
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")
                    # 修改点2：四舍五入并转为整数
                    xmin = round(float(bndbox.find("xmin").text))
                    ymin = round(float(bndbox.find("ymin").text))
                    xmax = round(float(bndbox.find("xmax").text))
                    ymax = round(float(bndbox.find("ymax").text))
                    
                    # 确保坐标有效性
                    xmin = max(0, min(xmin, width-1))
                    ymin = max(0, min(ymin, height-1))
                    xmax = max(0, min(xmax, width))
                    ymax = max(0, min(ymax, height))
                    
                    # 跳过无效标注
                    if xmin >= xmax or ymin >= ymax:
                        continue

                    width_box = xmax - xmin
                    height_box = ymax - ymin
                    
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id_map[obj.find("name").text],
                        "bbox": [xmin, ymin, width_box, height_box],  # 整数坐标
                        "area": width_box * height_box,
                        "iscrowd": 0
                    })
                    ann_id += 1

        # 保存标注文件
        with open(f"{output_dir}/annotations/instances_{dst_folder}.json", "w") as f:
            json.dump(coco_data, f, indent=2)

    print(f"COCO数据集已生成到 {output_dir}")

# 使用示例
if __name__ == "__main__":
    # VOC转COCO
    voc_to_coco("./VOC2007","./voc2coco")