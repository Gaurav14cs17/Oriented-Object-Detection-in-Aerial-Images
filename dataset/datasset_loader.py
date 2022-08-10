from .base import BaseDataset
import os
import cv2
import numpy as np


class DOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(DOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['plane',
                         'baseball-diamond',
                         'bridge',
                         'ground-track-field',
                         'small-vehicle',
                         'large-vehicle',
                         'ship',
                         'tennis-court',
                         'basketball-court',
                         'storage-tank',
                         'soccer-ball-field',
                         'roundabout',
                         'harbor',
                         'swimming-pool',
                         'helicopter'
                         ]
        self.color_pans = [(204, 78, 210),
                           (0, 192, 255),
                           (0, 131, 0),
                           (240, 176, 0),
                           (254, 100, 38),
                           (0, 0, 255),
                           (182, 117, 46),
                           (185, 60, 129),
                           (204, 153, 255),
                           (80, 208, 146),
                           (0, 0, 204),
                           (17, 90, 197),
                           (0, 255, 255),
                           (102, 255, 102),
                           (255, 255, 0)]

        self.num_classes = len(self.category)
        self.cat_ids = {cat: i for i, cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        if self.phase == 'train':
            image_set_index_file_path = os.path.join(self.data_dir, 'trainval.txt')
        else:
            image_set_index_file_path = os.path.join(self.data_dir, self.phase + '.txt')

        assert os.path.exists(image_set_index_file_path), 'Path does not exist :{}'.format(image_set_index_file_path)
        with open(image_set_index_file_path, 'r') as f:
            lines = f.readlines()

        images_list = [line.strip() for line in lines]
        return images_list

    def load_image(self, index):
        image_id = self.img_ids[index]
        image_file_path = os.path.join(self.image_path, image_id + '.png')
        assert os.path.exists(image_file_path), "image {} not existed".format(image_file_path)
        return cv2.imread(image_file_path)

    def load_annoFolder(self, image_id):
        ann_txt_path = os.path.join(self.label_path, image_id + ".txt")
        assert os.path.exists(ann_txt_path), "Ann file {} not existed".format(ann_txt_path)
        return ann_txt_path

    def get_valid_pts(self, x, y, w, h):
        x1 = min(max(float(x), 0), w - 1)
        y1 = min(max(float(y), 0), h - 1)
        return x1, y1

    def load_annotation(self, index):
        image = self.load_image(index)
        h, w, c = image.shape

        valid_pts = []
        valid_cat = []
        valid_dif = []

        ann_txt_path = self.load_annoFolder(index)
        with open(ann_txt_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                obj = line.split(' ')
                if len(obj) > 8:
                    x1, y1 = self.get_valid_pts(obj[0], obj[1], w, h)
                    x2, y2 = self.get_valid_pts(obj[2], obj[3], w, h)
                    x3, y3 = self.get_valid_pts(obj[4], obj[5], w, h)
                    x4, y4 = self.get_valid_pts(obj[6], obj[7], w, h)

                    # filtered smallest object 10x10
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)

                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                        valid_pts.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        valid_cat.append(self.cat_ids[obj[8]])
                        valid_dif.append(int(obj[9]))

        f.close()
        annotation = {'pts': np.asarray(valid_pts, np.float32), 'cat': np.asarray(valid_cat, np.int32),
                      'dif': np.asarray(valid_dif, np.int32)}
        return annotation





