"""
Name : classification_net.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-17 14:26
Desc:
"""

from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim
from libs.resnet_v1 import *
from libs.data_preprocess import *
import numpy as np
import os


class ClassificationNet:
    """
    cnn model
    """

    def __init__(self, sess,
                 backbones,
                 pretrained_model,
                 height=None,
                 width=None,
                 channels=3,
                 class_num=109,
                 train_backbone=True,
                 data_perprocess_op=None):
        """

        :param sess:
        :param backbones:
        :param pretrained_model:
        :param height:
        :param width:
        :param channels:
        :param class_num:
        """
        self.sess = sess
        self.images = tf.placeholder(tf.float32, shape=[None, height, width, channels])
        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.width = width
        self.height = height
        self.backbones = backbones
        self.class_num = class_num
        self.data_perprocess_op = data_perprocess_op
        self.train_backbone = train_backbone
        self.pretrained_model = pretrained_model
        self.optimizer, self.loss, self.output, self.acc, self.softmax_loss_b = self.graph(learning_rate=0.001)
        if pretrained_model is not None:
            self.load_pretrained_model()

    def graph(self, learning_rate=0.001):
        """
        create a graph
        :param global_step:
        :param training_iters:
        :param learning_rate:
        :param decay_rate:
        :param momentum:
        :return:
        """
        net, end_points = inference(self.images, keep_probability=1, phase_train=self.train_backbone)
        logit = slim.fully_connected(net, self.class_num, activation_fn=None, scope='classification_node1', reuse=False)
        print("logit shape:", logit.get_shape())

        one_hot = tf.one_hot(self.labels, self.class_num)
        softmax_loss_b = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=logit)
        softmax_loss = tf.reduce_mean(softmax_loss_b)
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(one_hot, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(softmax_loss)  # , global_step=global_step)
        return optimizer, softmax_loss, logit, acc, softmax_loss_b

    def load_pretrained_model(self):
        """
        load_pretrained_model
        :return:
        """

        def get_variables_in_checkpoint_file(file_name):
            try:
                reader = pywrap_tensorflow.NewCheckpointReader(file_name)
                var_to_shape_map = reader.get_variable_to_shape_map()
                return var_to_shape_map
            except Exception as e:  # pylint: disable=broad-except
                print(str(e))
                if "corrupted compressed block contents" in str(e):
                    print("It's likely that your checkpoint file has been compressed "
                          "with SNAPPY.")

        def get_variables_to_restore(variables, var_keep_dic):
            variables_to_restore = []
            for v in variables:
                # exclude
                if v.name.split(':')[0] in var_keep_dic:
                    print('Variables restored: %s' % v.name)
                    variables_to_restore.append(v)
                else:
                    print('Variables restored: %s' % v.name)
            return variables_to_restore

        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables, name='init'))
        print("variables initilized ok")
        # Get dictionary of model variable
        var_keep_dic = get_variables_in_checkpoint_file(self.pretrained_model)
        # # Get the variables to restore
        variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, self.pretrained_model)

    def train(self, dataset_train, dataset_val, epochs, training_iters, val_interval=3000, val_iters=100, show_step=50, ckpt_path="./weight"):
        """
        train
        :param dataset_train:
        :param dataset_val:
        :param epochs:
        :param training_iters:
        :param val_interval:
        :param val_iters:
        :param show_step:
        :param ckpt_path:
        :return:
        """
        fp1 = open("hard.txt","w+")
        fp = open("labels.txt",'r')
        lines = fp.readlines()
        labels = [int(label.split(':')[0]) for label in lines]
        names = [label.split(':')[1] for label in lines]
        print("Start Training")
        iterator_train = dataset_train.make_initializable_iterator()
        init_op_train = iterator_train.make_initializer(dataset_train)
        iterator_val = dataset_val.make_initializable_iterator()
        init_op_val = iterator_val.make_initializer(dataset_val)
        self.sess.run([init_op_train, init_op_val])

        iterator_train = iterator_train.get_next()
        iterator_val = iterator_val.get_next()

        saver = tf.train.Saver(max_to_keep=4)
        coord = tf.train.Coordinator()
        for epoch in range(epochs):
            for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                batch_x, batch_y = self.sess.run(iterator_train)
                loss = self.sess.run(self.loss, feed_dict={self.images: batch_x,
                                                           self.labels: batch_y
                                                           })

                # validation on training
                # if step % val_interval == 0 and step >= val_interval:
                #     accuarys = 0.0
                #     losses = 0.0
                #     for i in range(val_iters):
                #         val_batch_x, val_batch_y = self.sess.run(iterator_val)
                #         acc, loss = self.sess.run([self.acc, self.loss], feed_dict={self.images: val_batch_x, self.labels: val_batch_y})
                #         accuarys += acc
                #         losses += loss
                #     print("Accuary: {}, Loss: {}".format((accuarys / val_iters), (losses / val_iters)))
                if step % show_step == 0 and step > 0:
                    if loss > 1:
                        softmax_loss_b = self.sess.run(self.softmax_loss_b, feed_dict={self.images: batch_x,
                                                                                       self.labels: batch_y
                                                                                       })
                        for i in range(softmax_loss_b.shape[0]):
                            if softmax_loss_b[i] > 1:
                                index = labels.index(batch_y[i])
                                logs = 'label: {}, name: {}'.format(batch_y[i], names[index])
                                print(logs)
                                fp1.write(logs + "\n")
                    print("epoch: {} , step: {} , Loss: {}".format(epoch, step, loss))
                self.sess.run(self.optimizer, feed_dict={self.images: batch_x,
                                                         self.labels: batch_y
                                                         })
            ckpt_name = self.backbones + '_' + str(epoch) + '.ckpt'
            saver.save(self.sess, os.path.join(ckpt_path, ckpt_name))
        coord.request_stop()

    def forward(self, image):
        image = data_perprocess(image, self.data_perprocess_op)
        image = np.expand_dims(image, axis=0)
        output = self.sess.run([self.output], feed_dict={self.images: image})
        labels = np.argsort(output)[:, ::-1]
        labels = labels[0, 0:10]
        print(labels)
        return labels
