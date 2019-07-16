"""
Name : run.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-16 12:17
Desc:
"""

from trainer.train_tripletloss import trainer

height = 512
width = 512
people_per_batch = 15
images_per_person = 1000
batch_size = 60
epoch_size = 1000
embedding_size = 1024

trainer(height=512, width=512, people_per_batch=30, images_per_person=1000, batch_size=60, epoch_size=1000, embedding_size=1024)
