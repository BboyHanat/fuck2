# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
from network import facenet
import lfw

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange


def main(parameter_dict):
    network = importlib.import_module(parameter_dict['model_def'])

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(parameter_dict['logs_base_dir']), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(parameter_dict['models_base_dir']), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=parameter_dict['seed'])
    train_set = facenet.get_dataset(parameter_dict['data_dir'])

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if parameter_dict['pretrained_model']:
        print('Pre-trained model: %s' % os.path.expanduser(parameter_dict['pretrained_model']))

    # if parameter_dict['lfw_dir']:
    #     print('LFW directory: %s' % parameter_dict['lfw_dir'])
    #     # Read the file containing the pairs used for testing
    #     pairs = lfw.read_pairs(os.path.expanduser(parameter_dict['lfw_pairs']))
    #     # Get the paths for the corresponding images
    #     lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(parameter_dict['lfw_dir']), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(parameter_dict['seed'])
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                image = tf.image.resize_image_with_pad(image, parameter_dict['height'], parameter_dict['width'])

                # pylint: disable=no-member
                image.set_shape((parameter_dict['height'], parameter_dict['width'], 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(parameter_dict['height'], parameter_dict['width'], 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * parameter_dict['batch_size'],
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, parameter_dict['keep_probability'],
                                         phase_train=True, bottleneck_layer_size=parameter_dict['embedding_size'],
                                         weight_decay=parameter_dict['weight_decay'])

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, parameter_dict['embedding_size']]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, parameter_dict['alpha'])

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   parameter_dict['learning_rate_decay_epochs'] * parameter_dict['epoch_size'], parameter_dict['learning_rate_decay_factor'], staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, parameter_dict['optimizer'],
                                 learning_rate, parameter_dict['moving_average_decay'], tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=parameter_dict['gpu_memory_fraction'])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if parameter_dict['pretrained_model']:
                print('Restoring pretrained model: %s' % parameter_dict['pretrained_model'])
                model_path = os.path.expanduser(parameter_dict['pretrained_model'])
                facenet.load_pretrained_model(sess,model_path)

            # Training and validation loop
            epoch = 0
            while epoch < parameter_dict['max_nrof_epochs']:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // parameter_dict['epoch_size']
                # Train for one epoch
                train(parameter_dict, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                      batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step,
                      embeddings, total_loss, train_op, summary_op, summary_writer, parameter_dict['learning_rate_schedule_file'],
                      parameter_dict['embedding_size'], anchor, positive, negative, triplet_loss)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # # Evaluate on LFW
                # if args.lfw_dir:
                #     evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                #              batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, args.batch_size,
                #              args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)

    return model_dir


def train(parameter_dict, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0

    if parameter_dict['learning_rate'] > 0.0:
        lr = parameter_dict['learning_rate']
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < parameter_dict['epoch_size']:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, parameter_dict['people_per_batch'], parameter_dict['images_per_person'])

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = parameter_dict['people_per_batch'] * parameter_dict['images_per_person']
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / parameter_dict['batch_size']))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples - i * parameter_dict['batch_size'], parameter_dict['batch_size'])
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                       learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab, :] = emb
        print('%.3f' % (time.time() - start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    image_paths, parameter_dict['people_per_batch'], parameter_dict['alpha'])
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets * 3 / parameter_dict['batch_size']))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * parameter_dict['batch_size'], parameter_dict['batch_size'])
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, parameter_dict['epoch_size'], duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        # Add validation loss and accuracy to summary
        # pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                       (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size,
             nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                   learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=512)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=35)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=1000)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


def trainer(height=512,
            width=512,
            people_per_batch=45,
            images_per_person=40,
            batch_size=90,
            epoch_size=1000,
            embedding_size=1024):
    """
    训练入口函数
    :param height:
    :param width:
    :param people_per_batch:
    :param images_per_person:
    :param batch_size:
    :param epoch_size:
    :param embedding_size:
    :return:
    """
    training_parm = dict(
        data_dir="./dataset/train_fonts29/train",  # 训练数据的路径
        channels=3,
        class_num=29,
        logs_base_dir='/logs/facenet',
        models_base_dir='./weight/train/triplet',
        pretrained_model='./weight/pretrained/resnet_v1_50.ckpt',
        model_def='libs.resnet_v1',
        gpu_memory_fraction=1.0,
        height=height,
        width=width,
        max_nrof_epochs=64,
        people_per_batch=people_per_batch,  # 1 个 batch_size 要有 people_per_batch 个人的训练样本
        images_per_person=images_per_person,  # 每个人有多少个样本
        batch_size=batch_size,  # 一个 batch_size 要有 batch_size 个训练样本
        epoch_size=epoch_size,  # 一个 epoch 要有 epoch_size 张训练样本
        embedding_size=embedding_size,  # 输出向量的维度
        optimizer='ADAGRAD',  # ['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']
        keep_probability=1.0,
        weight_decay=0.0,
        alpha=0.5,
        learning_rate=0.1,
        learning_rate_schedule_file='dataset/learning_rate_retrain_tripletloss.txt',
        learning_rate_decay_epochs=20,
        learning_rate_decay_factor=1,
        moving_average_decay=0.9999,
        seed=666
    )
    main(training_parm)
