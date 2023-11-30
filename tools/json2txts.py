import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def read_img(img_dict):
    img_path = img_dict.get('file', '')
    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = np.array(img)[..., ::-1]
            img = np.ascontiguousarray(img)

    return img


def _json2txts(img_dict, txt_dir=None, conf=0.3):
    if img_dict.get('failure', False):
        return
    if not img_dict.get('detections', None):
        return

    max_conf = max(d['conf'] for d in img_dict['detections'])
    if max_conf < conf:
        return

    classes = [int(d['category']) - 1
               for d in img_dict['detections'] if d['conf'] > conf]
    boxes = np.array([d['bbox']
                      for d in img_dict['detections'] if d['conf'] > conf])
    # x1y1 -> xcyc
    boxes[:, 0] += boxes[:, 2] / 2
    boxes[:, 1] += boxes[:, 3] / 2

    txt_path = txt_dir / f"{Path(img_dict['file']).stem}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        for c, b in zip(classes, boxes):
            f.write(f'{c} {b[0]} {b[1]} {b[2]} {b[3]}\n')


def json2txts(cwd, conf=0.3):
    cwd = Path(r'U:\Animal\Private\test_feedback\20231123\RecM05_20231120_155641_155702_SW_671ECA000_48C76F')
    txt_dir = cwd / 'labels'
    txt_dir.mkdir(exist_ok=True)

    json_path = cwd / 'md.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    func = functools.partial(_json2txts, txt_dir=txt_dir, conf=conf)
    with ThreadPoolExecutor(8) as executor:
        list(tqdm(executor.map(func, json_data['images']),
                  total=len(json_data['images'])))


start_fmt = r'''<annotation>
    <folder>images</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
'''
obj_fmt = r'''    <object>
        <name>{}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
'''
end_fmt = '''</annotation>
'''


def txts2xmls(cwd):
    """Transform txts in the cwd into xmls"""
    # Glob txts
    txt_paths = sorted(Path(cwd).glob('labels/*.txt'))
    s = os.sep  # '/' in Linux, '\\' in Windows
    for txt_path in tqdm(txt_paths):
        # Get xml path
        xml_path = txt_path.with_suffix('.xml')
        xml_path = str(xml_path).replace(f'{s}labels{s}', f'{s}labels_xml{s}')
        xml_path = Path(xml_path)
        if xml_path.exists():
            continue

        # Read txt
        with open(txt_path, 'r', encoding='utf-8') as f:
            labels = np.array([list(map(eval, line.split())) for line in f])
        if labels.size == 0:
            labels = np.zeros((1, 5)) - 1  # Compatibility with empty images
        classes, boxes = labels[:, 0].astype(int), labels[:, 1:]

        # (xc, yc, w, h) norm -> (xmin, ymin, xmax, ymax) norm
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Read the image
        img_path = str(txt_path).replace(f'{s}labels{s}', f'{s}images{s}')
        img_path = Path(img_path).with_suffix('.jpg')
        if not img_path.exists():
            img_path = img_path.with_suffix('.png')
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
        else:
            with Image.open(img_path) as img:
                w, h = img.size

        # norm -> pixel
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = boxes.astype(int)

        # Write in xml
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(start_fmt.format(img_path.name, str(img_path), w, h))
            for class_id, box in zip(classes, boxes):
                if class_id == -1:
                    continue
                c = ['animal', 'person', 'vehicle'][class_id]
                f.write(obj_fmt.format(c, *box))
            f.write(end_fmt)


def main():
    cwd = Path(r'U:\Animal\Private\test_feedback\20231123\RecM05_20231120_155641_155702_SW_671ECA000_48C76F')
    json2txts(cwd)
    txts2xmls(cwd)


if __name__ == '__main__':
    main()
