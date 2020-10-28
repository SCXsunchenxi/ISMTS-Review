from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pickle
import argparse
import numpy as np
import tensorflow as tf
import os
from models import SGAN


FLAGS = None
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(_):
    tmp = pickle.load(open("data/cifar10_train.pkl", "rb"))
    x_train = tmp['data'].astype(np.float32).reshape([-1, 32, 32, 3]) / 127.5 - 1.
    print(x_train.shape)
    model = SGAN(
        num_z=FLAGS.num_z,
        beta=FLAGS.beta,
        num_gens=FLAGS.num_gens,
        d_batch_size=FLAGS.d_batch_size,
        g_batch_size=FLAGS.g_batch_size,
        z_prior=FLAGS.z_prior,
        learning_rate1=FLAGS.learning_rate1,
        learning_rate2=FLAGS.learning_rate2,
        img_size=(32, 32, 3),
        g_num_conv_layers=FLAGS.g_num_conv_layers,
        d_num_conv_layers=FLAGS.d_num_conv_layers,
        num_gen_feature_maps=FLAGS.num_gen_feature_maps,
        num_dis_feature_maps=FLAGS.num_dis_feature_maps,
        num_epochs=FLAGS.num_epochs,
        sample_fp="classifier_samples_g"+str(FLAGS.num_gens)+"_epoch_"+str(FLAGS.num_epochs)+"_layers_"+str(FLAGS.g_num_conv_layers)+"_lr1_"+str(FLAGS.learning_rate1)+"_lr2_"+str(FLAGS.learning_rate2)+"/samples_{epoch:04d}.png", 
        sample_by_gen_fp="classifier_samples_by_gen_g"+str(FLAGS.num_gens)+"_epoch_"+str(FLAGS.num_epochs)+"_layers_"+str(FLAGS.g_num_conv_layers)+"_lr1_"+str(FLAGS.learning_rate1)+"_lr2_"+str(FLAGS.learning_rate2)+"_new",
        random_seed=6789,
        checkpoint_dir="classifier_checkpoint_g"+str(FLAGS.num_gens)+"_best")
    model.fit(x_train)
    # model.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_z', type=int, default=100,
                        help='Number of latent units.')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Diversity parameter beta.')
    parser.add_argument('--num_gens', type=int, default=10,
                        help='Number of generators.')
    parser.add_argument('--d_batch_size', type=int, default=64,
                        help='Minibatch size for the discriminator.')
    parser.add_argument('--g_batch_size', type=int, default=64,
                        help='Minibatch size for the generators.')
    parser.add_argument('--z_prior', type=str, default="uniform",
                        help='Prior distribution of the noise (uniform/gaussian).')
    parser.add_argument('--learning_rate1', type=float, default=0.000005,
                        help='Learning rate1.')
    parser.add_argument('--learning_rate2', type=float, default=0.00001,
                        help='Learning rate2.')
    parser.add_argument('--g_num_conv_layers', type=int, default=2,
                        help='Number of G convolutional layers.')
    parser.add_argument('--d_num_conv_layers', type=int, default=3,
                        help='Number of D convolutional layers.')
    parser.add_argument('--num_gen_feature_maps', type=int, default=128,
                        help='Number of feature maps of Generator.')
    parser.add_argument('--num_dis_feature_maps', type=int, default=128,
                        help='Number of feature maps of Discriminator.')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
