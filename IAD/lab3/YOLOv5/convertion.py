import os
import xml.etree.ElementTree as ET

classes = ["trafficlight", 'stop', 'speedlimit', 'crosswalk']

images_folder = "./images"
annotations_folder = "./annotations"
output_folder = "./yolo"

os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)

def convert_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

for annotation_file in os.listdir(annotations_folder):
    if annotation_file.endswith(".xml"):
        # Parse XML file
        xml_path = os.path.join(annotations_folder, annotation_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image size
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Create YOLO label file
        label_file = os.path.join(output_folder, "labels", annotation_file.replace(".xml", ".txt"))
        with open(label_file, "w") as f:
            for obj in root.iter("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    continue
                class_id = classes.index(class_name)
                xmlbox = obj.find("bndbox")
                b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
                     float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
                yolo_box = convert_to_yolo((width, height), b)
                f.write(f"{class_id} " + " ".join([f"{a:.6f}" for a in yolo_box]) + "\n")

        # Copy image to output folder
        image_name = root.find("filename").text
        image_path = os.path.join(images_folder, image_name)
        output_image_path = os.path.join(output_folder, "images", image_name)
        if os.path.exists(image_path):
            os.system(f"cp {image_path} {output_image_path}")