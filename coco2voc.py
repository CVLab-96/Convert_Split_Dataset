import os
import json
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

def coco_to_voc(coco_root, output_dir="VOCDataset"):
    # 创建VOC目录结构
    os.makedirs(f"{output_dir}/JPEGImages", exist_ok=True)
    os.makedirs(f"{output_dir}/Annotations", exist_ok=True)
    os.makedirs(f"{output_dir}/ImageSets/Main", exist_ok=True)

    # 获取父文件夹名称
    folder_name = os.path.basename(os.path.normpath(output_dir))
    
    # 加载COCO标注
    categories = {}
    for split in ["train", "val"]:
        ann_file = f"{coco_root}/annotations/instances_{split}2014.json"
        if not os.path.exists(ann_file):
            continue

        with open(ann_file) as f:
            data = json.load(f)
        
        # 建立类别映射
        categories.update({cat["id"]: cat["name"] for cat in data["categories"]})

        # 处理图片和标注
        image_sets = []
        for img in tqdm(data["images"], desc=f"Processing {split} set"):
            # 复制图片
            src_img = f"{coco_root}/images/{split}2014/{img['file_name']}"
            dst_img = f"{output_dir}/JPEGImages/{img['file_name']}"
            shutil.copy(src_img, dst_img)
            image_sets.append(os.path.splitext(img['file_name'])[0])

            # 创建XML标注
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "folder").text = folder_name  # 修改点1：使用父文件夹名称
            ET.SubElement(annotation, "filename").text = img['file_name']
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(img['width'])
            ET.SubElement(size, "height").text = str(img['height'])
            ET.SubElement(size, "depth").text = "3"

            # 添加标注信息
            for ann in data["annotations"]:
                if ann["image_id"] == img["id"]:
                    # 转换坐标并四舍五入为整数
                    bbox = ann["bbox"]
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2]
                    h = bbox[3]
                    
                    # 计算边界坐标
                    xmin = max(0, round(x))
                    ymin = max(0, round(y))
                    xmax = min(img['width'], round(x + w))
                    ymax = min(img['height'], round(y + h))

                    # 跳过无效标注
                    if xmin >= xmax or ymin >= ymax:
                        continue

                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = categories[ann["category_id"]]
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    
                    bndbox = ET.SubElement(obj, "bndbox")
                    # 修改点2：使用整数坐标
                    ET.SubElement(bndbox, "xmin").text = str(xmin)
                    ET.SubElement(bndbox, "ymin").text = str(ymin)
                    ET.SubElement(bndbox, "xmax").text = str(xmax)
                    ET.SubElement(bndbox, "ymax").text = str(ymax)

            # 保存XML文件
            xml_str = ET.tostring(annotation, 'utf-8')
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
            xml_path = f"{output_dir}/Annotations/{os.path.splitext(img['file_name'])[0]}.xml"
            with open(xml_path, "w") as f:
                f.write(pretty_xml)

        # 保存图像集文件
        with open(f"{output_dir}/ImageSets/Main/{split}.txt", "w") as f:
            f.write("\n".join(image_sets))

    # 生成classes.txt
    with open(f"{output_dir}/classes.txt", "w") as f:
        unique_categories = sorted(set(categories.values()), 
                                key=lambda x: list(categories.values()).index(x))
        f.write("\n".join(unique_categories))

    print(f"VOC数据集已生成到 {output_dir}")


# 使用示例
if __name__ == "__main__":
    # COCO转VOC
    coco_to_voc("./yolo2coco","./coco2voc")