"""
Name : sample_select.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-19 10:41
Desc:
"""


import os
import shutil

path1 = "./dataset/train_fonts99/train_hard"
path2 = "./dataset/train_fonts99/val"
path3 = "./dataset/train_fonts99/val_hard"

class_name1 = [dir_class for dir_class in os.listdir(path1)]
class_name2 = [os.path.join(path2, dir_class) for dir_class in os.listdir(path2)]

for name in class_name2:
    if name.split("/")[-1] in class_name1:
        move_path = path3+"/"+name.split("/")[-1]
        shutil.move(name, move_path)