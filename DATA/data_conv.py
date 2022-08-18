import os
import pandas as pd
import json, cv2
import numpy as np

csv_path = "./Tooth_restoration_csv (1).csv"
images_path = "tooth_2&8"
output_path = "/media/gaurav/DATA/labs/teeth_dental/Annotation_tooth_restoration/output_images"

'''
x1, y1, x2, y2, x3, y3, x4, y4, category, difficulty

'''


def draw_output(ori_image, all_x_pts, all_y_pts, class_name, f):
    tl = np.asarray([all_x_pts[0], all_y_pts[0]], np.float32)
    tr = np.asarray([all_x_pts[1], all_y_pts[1]], np.float32)
    br = np.asarray([all_x_pts[2], all_y_pts[2]], np.float32)
    bl = np.asarray([all_x_pts[3], all_y_pts[3]], np.float32)

    tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
    rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
    bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
    ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

    box = np.asarray([tl, tr, br, bl], np.float32)

    cen_pts = np.mean(box, axis=0)
    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 1, 1)
    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)

    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tl[0]), int(tl[1])), (0,0,255),1,1)
    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tr[0]), int(tr[1])), (255,0,255),1,1)
    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(br[0]), int(br[1])), (0,255,0),1,1)
    # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bl[0]), int(bl[1])), (255,0,0),1,1)
    ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (255, 0, 255), 1, 1)
    # box = cv2.boxPoints(cv2.minAreaRect(box))
    # ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (0,255,0),1,1)
    class_id = 0 if class_name == "Tooth" else 1

    cv2.putText(ori_image, '{}'.format(class_name), (int(box[1][0]), int(box[1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (0, 255, 255), 1, 1)

    f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {} {} \n'.format(all_x_pts[0], all_y_pts[0],
                                                                                      all_x_pts[1],
                                                                                      all_y_pts[1], all_x_pts[2],
                                                                                      all_y_pts[2],
                                                                                      all_x_pts[3], all_y_pts[3],
                                                                                      class_name, class_id))
    return ori_image


data = pd.read_csv(csv_path)
prev_image = None
prev_name = None
txt_outpath = "./txt_output/"

if not os.path.exists(txt_outpath):
    os.mkdir(txt_outpath)

for row in zip(data["filename"], data['region_id'], data['region_shape_attributes'], data['region_attributes']):
    try:
        file_name = row[0]
        image_id = int(row[1])
        all_x_pts = json.loads(row[2])['all_points_x']
        all_y_pts = json.loads(row[2])['all_points_y']
        class_name = json.loads(row[3])['shape']
        if image_id == 0:
            image = cv2.imread(os.path.join(images_path, file_name))
            txt_outpath_ = os.path.join(txt_outpath, file_name.split('.')[0] + '.txt')
            f = open(os.path.join(txt_outpath_), 'w')
            # f.write("imagesource:GoogleEarth \n")
            # f.write("gsd:0.146343590398 \n")

        image = draw_output(image, all_x_pts, all_y_pts, class_name, f)

        if image_id == 0 and prev_image is not None:
            output_path_image_name = os.path.join(output_path, prev_name)
            cv2.imwrite(output_path_image_name, prev_image)
            print(output_path_image_name)
            # f.close()

        prev_image = image.copy()
        prev_name = file_name

    except  Exception as e:
        print(e)
