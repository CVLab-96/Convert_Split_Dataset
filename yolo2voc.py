import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
import cv2

def yolo_to_voc(yolo_dataset_path, voc_dataset_path):
    # 创建输出目录
    os.makedirs(os.path.join(voc_dataset_path, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(voc_dataset_path, "Annotations"), exist_ok=True)
    os.makedirs(os.path.join(voc_dataset_path, "ImageSets", "Main"), exist_ok=True)

    # 获取所有类别
    classes = set()
    for split in ["train", "val", "test"]:
        split_path = os.path.join(yolo_dataset_path, split)
        if not os.path.exists(split_path):
            continue
        
        labels_path = os.path.join(split_path, "labels")
        if not os.path.exists(labels_path):
            continue
        
        for label_file in os.listdir(labels_path):
            if not label_file.endswith(".txt"):
                continue
            with open(os.path.join(labels_path, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        classes.add(parts[0])
    
    # 将类别排序并写入classes.txt
    classes = sorted(list(classes))
    with open(os.path.join(voc_dataset_path, "classes.txt"), "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Generated classes.txt with {len(classes)} classes.")

    # 转换每个split
    for split in ["train", "val", "test"]:
        split_path = os.path.join(yolo_dataset_path, split)
        if not os.path.exists(split_path):
            continue
        
        images_path = os.path.join(split_path, "images")
        labels_path = os.path.join(split_path, "labels")
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            continue
        
        # 创建ImageSets文件
        with open(os.path.join(voc_dataset_path, "ImageSets", "Main", f"{split}.txt"), "w") as f_set:
            for image_file in os.listdir(images_path):
                if not image_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                
                # 复制图片到JPEGImages
                image_name = os.path.splitext(image_file)[0]
                src_image_path = os.path.join(images_path, image_file)
                dst_image_path = os.path.join(voc_dataset_path, "JPEGImages", image_file)
                shutil.copy(src_image_path, dst_image_path)
                
                # 写入ImageSets
                f_set.write(f"{image_name}\n")
                
                # 生成XML
                label_file = os.path.join(labels_path, f"{image_name}.txt")
                if not os.path.exists(label_file):
                    continue
                
                # 读取图片尺寸
                img = cv2.imread(src_image_path)
                if img is None:
                    print(f"无法读取图片: {src_image_path}")
                    continue
                height, width, _ = img.shape
                
                # 创建XML结构
                annotation = ET.Element("annotation")
                ET.SubElement(annotation, "folder").text = os.path.basename(voc_dataset_path)
                ET.SubElement(annotation, "filename").text = image_file
                
                source = ET.SubElement(annotation, "source")
                ET.SubElement(source, "database").text = "The VOC2007 Database"
                ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
                ET.SubElement(source, "image").text = "flickr"
                ET.SubElement(source, "flickrid").text = "325991873"
                
                owner = ET.SubElement(annotation, "owner")
                ET.SubElement(owner, "flickrid").text = "archintent louisville"
                ET.SubElement(owner, "name").text = "?"
                
                size = ET.SubElement(annotation, "size")
                ET.SubElement(size, "width").text = str(width)
                ET.SubElement(size, "height").text = str(height)
                ET.SubElement(size, "depth").text = str(3)
                
                ET.SubElement(annotation, "segmented").text = "0"
                
                # 读取标签并转换为VOC格式
                with open(label_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = parts[0]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width_yolo = float(parts[3])
                        height_yolo = float(parts[4])
                        
                        # 转换为VOC格式 (xmin, ymin, xmax, ymax)
                        xmin = max(0, round((x_center - width_yolo / 2) * width))
                        ymin = max(0, round((y_center - height_yolo / 2) * height))
                        xmax = min(width, round((x_center + width_yolo / 2) * width))
                        ymax = min(height, round((y_center + height_yolo / 2) * height))
                        
                        # 添加到XML
                        obj = ET.SubElement(annotation, "object")
                        ET.SubElement(obj, "name").text = class_id
                        ET.SubElement(obj, "pose").text = "Unspecified"
                        ET.SubElement(obj, "truncated").text = "0"
                        ET.SubElement(obj, "difficult").text = "0"
                        
                        bndbox = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bndbox, "xmin").text = str(xmin)
                        ET.SubElement(bndbox, "ymin").text = str(ymin)
                        ET.SubElement(bndbox, "xmax").text = str(xmax)
                        ET.SubElement(bndbox, "ymax").text = str(ymax)
                
                # 生成格式化的XML
                xml_str = ET.tostring(annotation)
                xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")
                
                # 写入XML文件
                xml_path = os.path.join(voc_dataset_path, "Annotations", f"{image_name}.xml")
                with open(xml_path, "w") as xml_file:
                    xml_file.write(xml_str)
                
                print(f"Converted {image_name} to VOC format. Image copied to {dst_image_path}")

    print("Conversion completed.")

if __name__ == "__main__":
    yolo_dataset_path = input("请输入YOLO数据集路径: ")
    voc_dataset_path = input("请输入输出VOC数据集路径: ")
    yolo_to_voc(yolo_dataset_path, voc_dataset_path)