from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import json


class PascalVOCDataset(Dataset):
    def __init__(self, data_dirs, data_list_file_name, imsize=448, grid_num=7, bbox_num=2, class_num=20, l_coord=5., l_noobj=0.5, transform=None):
        self.transform = transform
        self.data_paths = self._get_paths(data_dirs, data_list_file_name)
        self.imsize = imsize
        self.grid_num = grid_num
        self.bbox_num = bbox_num
        self.class_num = class_num
        self.label_map = self._get_label_map()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, label_path = self.data_paths[idx]

        image = Image.open(image_path).convert('RGB').resize((self.imsize, self.imsize))
        label, mask = self._get_label_mask(label_path)

        if self.transform:
            image = self.transform(image)

        return image, label, mask

    def _get_label_map(self):
        label_map_path = Path(__file__).parent / 'labelmap.json'
        with open(label_map_path, 'r') as f:
            labels = json.load(f)['PascalVOC']
        label_map = {label: i for i, label in enumerate(labels)}
        return label_map

    def _get_paths(self, data_dirs, data_list_file_name):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        paths = []
        for data_dir in data_dirs:
            data_list_path = Path(data_dir) / 'ImageSets' / 'Main' / data_list_file_name

            with open(data_list_path, 'r') as f:
                ids = f.read().split('\n')

            paths += [[Path(data_dir) / 'JPEGImages' / f'{i}.jpg', Path(data_dir) / 'Annotations' / f'{i}.xml'] for i in ids[:-1]]

        return paths

    def _get_label_mask(self, label_path):
        grid_len = self.imsize // self.grid_num

        root = ET.parse(label_path).getroot()
        size = root.find('size')
        width, height = int(size.find('width').text), int(size.find('height').text)
        label = torch.zeros((5 * self.bbox_num + self.class_num, self.grid_num, self.grid_num))
        mask = torch.zeros_like(label)

        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)
            xmin, xmax = self.imsize * xmin / width, self.imsize * xmax / width
            ymin, ymax = self.imsize * ymin / height, self.imsize * ymax / height
            c = self.label_map[obj.find('name').text]
            cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
            lx, ly = xmax - xmin, ymax - ymin
            idx, idy = int(cx // grid_len), int(cy // grid_len)
            for j in range(self.bbox_num):
                if label[5*j, idx, idy] == 0.:
                    label[0+5*j:2+5*j, idx, idy] = (torch.tensor([cx, cy]) % grid_len) / grid_len
                    label[2+5*j:4+5*j, idx, idy] = torch.sqrt((torch.tensor([lx, ly]) / self.imsize))
                    label[4+5*j, idx, idy] = 1.
                    mask[0+5*j:4+5*j, idx, idy] = self.l_coord
                    mask[4+5*j, idx, idy] = 1.
                    break
            for i in range(xmin // grid_len, (xmax // grid_len) + 1):
                for j in range(ymin // grid_len, (ymax // grid_len) + 1):
                    label[10+c, i, j] = 1.
                    mask[10:, i, j] = 1.

        for j in range(self.bbox_num):
            mask[4+5*j] += (1 - mask[4+5*j]) * self.l_noobj

        return label, mask
