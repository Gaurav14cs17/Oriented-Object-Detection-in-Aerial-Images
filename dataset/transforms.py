import numpy as np
import cv2


def random_flip(image, gt_pts, crop_center=None):
    h, w, c = image.shape

    # flip by x-axis...........
    if np.random.random() < 0.5:
        image = image[:, ::-1, :]
        if gt_pts.shape[0]:
            gt_pts[:, :, 0] = w - 1 - gt_pts[:, :, 0]
        if crop_center is not None:
            crop_center[0] = w - 1 - crop_center[0]

    # flip by y-axis...........
    if np.random.random() < 0.5:
        image = image[::-1, :, :]
        if gt_pts.shape[0]:
            gt_pts[:, :, 1] = h - 1 - gt_pts[:, :, 1]
        if crop_center is not None:
            crop_center[1] = h - 1 - crop_center[1]
    return image, gt_pts, crop_center


def _get_border(size, border):
    i = 1
    while (size - border) // i <= border // i:
        i = i * 2
    return border // i


def random_crop_info(h, w):
    if np.random.random() < 0.3:
        max_wh = max(h, w)
        random_size = max_wh * np.random.choice(np.arange(0.9, 1.1, 0.1))
        w_border = _get_border(size=w, border=32)
        h_border = _get_border(size=h, border=32)
        random_center_x = np.random.randint(low=w_border, high=w - w_border)
        random_center_y = np.random.randint(low=h_border, high=h - h_border)
        return [random_size, random_size], [random_center_x, random_center_y]
    else:
        return None, None


def Rotation_Transform(src_point, degree):
    radian = np.pi * degree / 180
    R_matrix = [[np.cos(radian), -np.sin(radian)],
                [np.sin(radian), np.cos(radian)]]

    R_matrix = np.array(R_matrix, np.float32)
    R_pts = np.matmul(R_matrix, src_point)
    return R_pts
