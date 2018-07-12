import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import numpy as np

def train(model_name, gpu_id):

    with tf.Session() as sess:


        params = param.get_general_params()

        network_dir = params['model_save_dir'] + '/' + model_name

        # Creates models directory if not exist.
        if not os.path.isdir(network_dir):
            print(network_dir)
            os.mkdir(network_dir)

        train_feed = data_generation.create_feed(params, params['data_dir'], 'train')

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        vgg_model = truncated_vgg.vgg_norm()
        networks.make_trainable(vgg_model, False)
        response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')
        model = networks.network_posewarp(params)
        model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

        n_iters = params['n_training_iter']

        summary_writer = tf.summary.FileWriter("D:\Proyectos\JEJU2018\Code\posewarp-cvpr2018\code\logs", graph=sess.graph)

        for step in range(0, n_iters):
            x, y = next(train_feed)

            train_loss = model.train_on_batch(x, y)

            util.printProgress(step, 0, train_loss)

            # out = sess.run(conv, feed_dict={"input_1:0" : x[0]})
            # plt.matshow(out[0, :, :, 0])
            # plt.show()

            conv = tf.get_default_graph().get_tensor_by_name("loss/add_2_loss/lambda_5/add:0")
            image_summary_op = tf.summary.image('images', conv[0:1, :, :, :], max_outputs=100)

            image_summary = sess.run(image_summary_op, feed_dict={"in_img0:0" : x[0], "in_pose0:0" : x[1], "in_pose1:0" : x[2], "mask_prior:0" : x[3], "trans_in:0" : x[4]})
            summary_writer.add_summary(image_summary)

            if step > 0 and step % params['model_save_interval'] == 0:
                model.save_weights(network_dir + '/' + str(step) + '.h5')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        train(sys.argv[1], sys.argv[2])
