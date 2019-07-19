"""
Name : run.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-16 12:17
Desc:
"""

from trainer import train_tripletloss, train_softmax

# train tripletloss
# train_tripletloss.trainer(height=512, width=512, people_per_batch=30, images_per_person=10, batch_size=60, epoch_size=1000, embedding_size=1024)

# train classification network
train_img_root = "./dataset/train_fonts99/train"
val_img_root = "./dataset/train_fonts99/val"
train_softmax.trainer(train_img_root, val_img_root, height=512, width=512, channels=3, class_num=90, train_batch_size=32, val_batch_size=16, epoch=64)
