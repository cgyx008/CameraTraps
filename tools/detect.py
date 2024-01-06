from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
import supervision as sv
import torch
from torch import nn
from tqdm import tqdm

from PytorchWildlife.models import detection as pw_detection


class Detector:
    def __init__(self, weights=None, device='cuda', pretrained=False):
        self.weights = weights or r'D:\Projects\CameraTraps\weights\v5\md_v5a.0.0.pt'
        self.device = device
        self.pretrained = pretrained
        self.model = self.get_model()

        self.img_paths = []
        self.img_path = Path('')
        self.img = None
        self.w, self.h = 0, 0
        self.empty_img_paths = []

        self.box_annotator = sv.BoxAnnotator()

    def get_model(self):
        detection_model = pw_detection.MegaDetectorV5(
            weights=self.weights,
            device=self.device,
            pretrained=self.pretrained
        )

        # Compatibility
        for m in detection_model.model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

        return detection_model

    def get_input_data(self):
        try:
            self.img = cv2.imread(str(self.img_path))
        except cv2.error as e:
            self.img = None
            print(self.img_path)
            print(e)
        if self.img is None:
            return None
        self.h, self.w = self.img.shape[:2]

        input_data = cv2.resize(self.img, (1280, 1280))
        input_data = (input_data[..., ::-1].transpose((2, 0, 1)) / 255)
        input_data = torch.from_numpy(input_data.astype(np.float32))
        return input_data

    def detect_img(self, input_data):
        detections = self.model.single_image_detection(input_data)
        return detections

    def scale_boxes(self, boxes):
        boxes_copy = boxes.copy()
        boxes_copy = boxes_copy.astype(float)
        boxes_copy[:, [0, 2]] *= self.w / 1280
        boxes_copy[:, [1, 3]] *= self.h / 1280
        boxes_copy = boxes_copy.astype(int)
        return boxes_copy

    def annotate_img(self, detections):
        annotated_img = self.box_annotator.annotate(
            scene=self.img,
            detections=detections["detections"],
            labels=detections["labels"]
        )
        return annotated_img

    def detect_img_dir(self, img_dir):
        img_paths = sorted(Path(img_dir).glob('**/images/*.jpg'))
        self.img_paths = img_paths
        self.make_xml_dirs()

        for img_path in tqdm(self.img_paths):
            self.img_path = img_path
            input_data = self.get_input_data()
            if input_data is None:
                self.empty_img_paths.append(img_path)
                continue
            detections = self.detect_img(input_data)
            detections['detections'].xyxy = self.scale_boxes(detections['detections'].xyxy)
            self.save_xml(detections)

    def make_xml_dirs(self):
        xml_dirs = {str(p.parent.parent / 'labels_xml') for p in self.img_paths}
        xml_dirs = sorted(list(xml_dirs))
        print('Making xml dirs...')
        for xml_dir in tqdm(xml_dirs):
            Path(xml_dir).mkdir(parents=True, exist_ok=True)

    def save_xml(self, detections=None):
        xml_dir = self.img_path.parent.parent / 'labels_xml'
        xml_path = xml_dir / self.img_path.with_suffix('.xml').name
        self.write_xml(xml_path, detections)

    def write_xml(self, xml_path, detections=None):
        # Create new xml
        root = ET.Element('annotation')
        # folder
        folder = ET.SubElement(root, 'folder')
        folder.text = self.img_path.parts[-2]
        # filename
        filename = ET.SubElement(root, 'filename')
        filename.text = self.img_path.name
        # path
        path = ET.SubElement(root, 'path')
        path.text = str(self.img_path)
        # source
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        # size
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(self.w)
        height = ET.SubElement(size, 'height')
        height.text = str(self.h)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        # segmented
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        # object
        if detections and detections.get('detections', None):
            labels = [c.split()[0] for c in detections['labels']]
            boxes = detections['detections'].xyxy
            for label, box in zip(labels, boxes):
                obj = ET.SubElement(root, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = label  # name
                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(box[0])
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(box[1])
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(box[2])
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(box[3])

        # Convert ElementTree to string
        xml_string = ET.tostring(root, encoding='utf-8')
        xml_string = ''.join(xml_string.decode('utf-8').split())

        # Parse the XML string using minidom
        dom = minidom.parseString(xml_string)
        pretty_xml_string = dom.toprettyxml()

        # Write the prettified XML string to a file
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml_string)


def main():
    detector = Detector()
    detector.detect_img_dir(r'U:\Bird\20240105YoutubeBirdTrain')
    assert 1


if __name__ == '__main__':
    main()
