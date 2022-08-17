import os

image_dir_path = "/media/gaurav/DATA/labs/Oriented-Object-Detection-in-Aerial-Images/DATA/images"
ann_txt_path = "trainval.txt"

with open(ann_txt_path, 'w') as f:
    for file_name in os.listdir(image_dir_path):
        txt_name = file_name.split('.jpg')[0] + "\n"
        f.write(txt_name)

