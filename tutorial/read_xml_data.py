from pathlib import Path
import xml.etree.ElementTree as ET

voc_2007_path = '/work/data/VOCdevkit/VOC2007'
train_path = Path(voc_2007_path) / 'ImageSets/Main/trainval.txt'

with open(train_path, 'r') as f:
    ids = f.read().split('\n')

sample = {}
for i in ids[:-1]:
    xml_path = Path(voc_2007_path) / 'Annotations' / f'{i}.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    break

sample['im_shape'] = [int(s.text) for s in root.find('size')]

sample['object'] = [{'class': obj.find('name').text, 'bbox': [int(b.text) for b in obj.find('bndbox')]} for obj in root.iter('object')]
print(sample)