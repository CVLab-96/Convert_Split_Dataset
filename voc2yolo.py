import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

def voc_to_yolo(voc_root, yolo_output):
    # 自动获取类别列表
    class_set = set()
    for xml_file in os.listdir(os.path.join(voc_root, "Annotations")):
        tree = ET.parse(os.path.join(voc_root, "Annotations", xml_file))
        for obj in tree.findall("object"):
            class_set.add(obj.find("name").text)
    class_list = sorted(list(class_set))
    with open(os.path.join(yolo_output, "classes.txt"), 'w') as f:
        f.write("\n".join(class_list))
    
    # 创建YOLO目录结构
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(yolo_output, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_output, split, "labels"), exist_ok=True)
    
    # 处理每个split
    for split in splits:
        with open(os.path.join(voc_root, "ImageSets/Main", f"{split}.txt")) as f:
            img_names = [line.strip() for line in f.readlines()]
        
        for img_name in tqdm(img_names, desc=f"Processing {split}"):
            # 复制图片
            src_img = os.path.join(voc_root, "JPEGImages", f"{img_name}.jpg")
            dst_img = os.path.join(yolo_output, split, "images", f"{img_name}.jpg")
            shutil.copy(src_img, dst_img)
            
            # 转换标签
            xml_path = os.path.join(voc_root, "Annotations", f"{img_name}.xml")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            w = int(root.find("size/width").text)
            h = int(root.find("size/height").text)
            
            txt_path = os.path.join(yolo_output, split, "labels", f"{img_name}.txt")
            with open(txt_path, 'w') as f:
                for obj in root.findall("object"):
                    cls = obj.find("name").text
                    class_id = class_list.index(cls)
                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)
                    
                    x_center = (xmin + xmax) / 2 / w
                    y_center = (ymin + ymax) / 2 / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

if __name__ == "__main__":
    voc_to_yolo(
        voc_root="./datasets/VOC2007",
        yolo_output="./voc2yolo"
    )