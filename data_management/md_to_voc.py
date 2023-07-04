import json
from xml.dom import minidom
import xml.etree.ElementTree as ET

import cv2
from pathlib import Path

from tqdm import tqdm


def init_voc_xml(img_path):
    # Read the image
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    h, w, c = img.shape

    # root
    root = ET.Element('annotation')
    # folder
    folder = ET.SubElement(root, 'folder')
    folder.text = img_path.parts[-2]
    # filename
    filename = ET.SubElement(root, 'filename')
    filename.text = img_path.name
    # path
    path = ET.SubElement(root, 'path')
    path.text = str(img_path)
    # source/database
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    # size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(c)
    # segmented
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'
    return root


def md_to_voc(md_path, voc_dir=None, min_conf=0.5):
    # Initialize arguments
    voc_dir = Path('.') if voc_dir is None else Path(voc_dir)
    voc_dir.mkdir(parents=True, exist_ok=True)

    # Read md json
    with open(md_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for img_dict in tqdm(json_data.get('images', [])):
        # Read image width and height
        img_path = Path(img_dict['file'])
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Initialize voc xml
        root = init_voc_xml(img_path)
        xml_path = Path(voc_dir) / f'{img_path.stem}.xml'

        if img_dict.get('max_detection_conf', 0) < min_conf:
            xmlstr = minidom.parseString(ET.tostring(root))
            xmlstr = xmlstr.toprettyxml(indent='\t')
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xmlstr)
            continue

        # Add object element
        for det_dict in img_dict.get('detections', []):
            if det_dict.get('conf', 0) < min_conf:
                continue
            # Only keep vehicle
            if det_dict['category'] != '3':
                continue
            # object
            obj = ET.SubElement(root, 'object')
            # name
            name = ET.SubElement(obj, 'name')
            cat_id = int(det_dict['category']) - 1
            name.text = ['Animal', 'Person', 'Vehicle'][cat_id]
            # pose
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            # truncated
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            # difficult
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            # bndbox
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')
            bbox = det_dict['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox[0] *= w
            bbox[1] *= h
            bbox[2] *= w
            bbox[3] *= h
            xmin.text = str(int(bbox[0]))
            ymin.text = str(int(bbox[1]))
            xmax.text = str(int(bbox[2]))
            ymax.text = str(int(bbox[3]))

        # Write xml
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent='\t')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xmlstr)


def main():
    md_to_voc(r'G:\Data\FEPD\Wall_mounted_9MP\md.json',
              r'G:\Data\FEPD\Wall_mounted_9MP\labels')


if __name__ == '__main__':
    main()
