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
import scipy.misc

def train(model_name, gpu_id):

    with tf.Session() as sess:

        params = param.get_general_params()

        network_dir = params['model_save_dir'] + '/' + model_name

        # Creates models directory if not exist.
        if not os.path.isdir(network_dir):
            os.mkdir(network_dir)

        train_feed = data_generation.create_feed(params, params['data_dir'], 'train')
        test_feed  = data_generation.create_feed(params, params['data_dir'], 'test')

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

        tr_x, tr_y = next(train_feed)
        te_x, te_y = next(test_feed)

        # Prepare output directories if they don't exist.
        output_dir = '../output/' + model_name + '/'

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        scipy.misc.imsave('../output/tr_orig_image.png', tr_x[0][0, :, :, :])
        scipy.misc.imsave('../output/tr_targ_image.png', tr_y[0, :, :, :])
        scipy.misc.imsave('../output/te_orig_image.png', te_x[0][0, :, :, :])
        scipy.misc.imsave('../output/te_targ_image.png', te_y[0, :, :, :])

        for step in range(0, n_iters):
            x, y = next(train_feed)

            train_loss = model.train_on_batch(x, y)

            util.printProgress(step, 0, train_loss)

            # out = sess.run(conv, feed_dict={"input_1:0" : x[0]})
            # plt.matshow(out[0, :, :, 0])
            # plt.show()

            gen = tf.get_default_graph().get_tensor_by_name("loss/add_2_loss/lambda_5/add:0")
            inp = tf.get_default_graph().get_tensor_by_name("in_img0:0")
            out = tf.get_default_graph().get_tensor_by_name("in_img1:0")
            p_s = tf.get_default_graph().get_tensor_by_name("mask_src/truediv:0")
            # p_t = tf.get_default_graph().get_tensor_by_name("in_pose1:0")

            image_summary_1 = tf.summary.image('images', [inp[0, :, :, :], out[0, :, :, :], gen[0, :, :, :]],
                                               max_outputs=100)
            # image_summary_2 = tf.summary.image('pose', [tf.reduce_sum(p_s[0, :, :, :], 2, keepdims=True)], max_outputs=100)

            image_summary_1 = sess.run(image_summary_1,
                                       feed_dict={"in_img0:0": x[0], "in_pose0:0": x[1], "in_pose1:0": x[2],
                                                  "mask_prior:0": x[3], "trans_in:0": x[4], "in_img1:0": y})

            # image_summary_2 = sess.run(image_summary_2, feed_dict={"in_img0:0" : x[0], "in_pose0:0" : x[1], "in_pose1:0" : x[2],
            #                                                     "mask_prior:0" : x[3], "trans_in:0" : x[4], "in_img1:0" : y})

            summary_writer.add_summary(image_summary_1)
            # summary_writer.add_summary(image_summary_2)

            train_image = sess.run(gen, feed_dict={"in_img0:0": tr_x[0], "in_pose0:0": tr_x[1], "in_pose1:0": tr_x[2],
                                                   "mask_prior:0": tr_x[3], "trans_in:0": tr_x[4], "in_img1:0": tr_y})

            test_image = sess.run(gen, feed_dict={"in_img0:0": te_x[0], "in_pose0:0": te_x[1], "in_pose1:0": te_x[2],
                                                  "mask_prior:0": te_x[3], "trans_in:0": te_x[4], "in_img1:0": te_y})
            if step > 0 and step % params['model_save_interval'] == 0:
                model.save_weights(network_dir + '/' + str(step) + '.h5')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        train(sys.argv[1], sys.argv[2])
