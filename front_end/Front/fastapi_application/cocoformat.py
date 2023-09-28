import os
import json
import xml.etree.ElementTree as ET

image_list = os.listdir("./")
image_list.remove("cocoformat.py")

data = {
    "images" : [] ,
    "annotations" : []
}

k = 1
for i in range(len(image_list)) :
    data["images"].append({"id": i+1, "file_name": image_list[i]})
    
    xml_file=image_list[i].replace("jpg","xml")
    xml_path = os.path.join("/home/moetez/Downloads/train_xml/annotation", xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tables=root.findall("object")
    for j in range(len(tables)) :
        xmax=int(((tables[j].find("bndbox")).find("xmax")).text)
        ymax=int(((tables[j].find("bndbox")).find("ymax")).text)
        xmin=int(((tables[j].find("bndbox")).find("xmin")).text)
        ymin=int(((tables[j].find("bndbox")).find("ymin")).text)
        data["annotations"].append({"id": k, "image_id": i+1, "category_id": 0, "iscrowd": 0, "area": (xmax-xmin)*(ymax-ymin) , "bbox": [xmin, ymin, xmax-xmin, ymax-ymin]})
        k+=1
with open("annotation.json", "w") as json_file:
    json.dump(data, json_file)

