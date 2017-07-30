import numpy as np
import tensorflow as tf

from utilities import PReLU, spatial_dropout, unpool

class ENet_model(object):
    """
    - DOES:
    """"

    def _init_(self, model_id):
        """
        - DOES:
        """

        self.model_id = model_id

        self.lr = 0.001

        self.logs_dir = "/home/fregu856/segmentation/training_logs"

        #
        self.create_model_dirs()
        #
        self.add_placeholders()
        #
        self.add_logits()
        #
        self.add_loss_op()
        #
        self.add_train_op()

    def create_model_dirs(self):
        """
        - DOES:
        """"

        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


    def add_placeholder(self):
        """
        - DOES:
        """

        self.imgs_ph = 0
        self.labels_ph = 0
        self.keep_prob_ph = 0
        self.training_ph = 0

    def create_feed_dict(self, imgs_batch, labels_batch=None, keep_prob=1, training=True):
        """
        - DOES: returns a feed_dict mapping the placeholders to the actual
        input data (this is how we run the network on specific data).
        """

        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.keep_prob_ph] = keep_prob
        feed_dict[self.training_ph] = training
        if labels_batch is not None:
            # only add the labels data if it's specified (during inference, we
            # won't have any labels):
            feed_dict[self.labels_ph] = labels_batch

    def add_logits(self):
        """
        - DOES:
        """

        # TODO!

        self.logits = 0

    def add_loss_op(self):
        """
        - DOES: computes the CE loss for the batch.
        """

        # TODO!

        self.loss = 0

    def add_train_op(self):
        """
        - DOES: creates a training operator for minimization of the loss.
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)



    def initial_block(self, x):
        with tf.variable_scope("initial_block"):
            W_conv = tf.get_variable("W",
                        shape=[3, 3, 3, 13], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            b_conv = tf.get_variable("b", shape=[13] # ([out_depth]], one bias weight per out depth layer),
                        initializer=tf.constant_initializer(0))

            conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                        padding="SAME") + b_conv

            pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding="VALID")

            concat = tf.concat([conv_branch, pool_branch], axis=3) # (3: the depth axis)

            output = tf.contrib.slim.batch_norm(net_conv,
                        is_training=self.training_ph)
            output = PReLU(output)

        return output

    def bottleneck_regular(self, x, output_depth, drop_prob, scope, proj_ratio=4, downsample=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(input_depth/proj_ratio)

        # convolution branch:
        # # 1x1 projection:
        if not downsample:
            W_proj = tf.get_variable(scope + "/W_proj",
                        shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            conv_branch = tf.nn.conv2d(x, W_proj, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
        else:
            W_conv = tf.get_variable(scope + "/W_proj",
                        shape=[2, 2, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                        padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch)

        # # conv:
        W_conv = tf.get_variable(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        b_conv = tf.get_variable(scope + "/b_conv", shape=[internal_depth] # ([out_depth]], one bias weight per out depth layer),
                    initializer=tf.constant_initializer(0))
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch)

        # # 1x1 expansion:
        W_exp = tf.get_variable(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob, training=self.training_ph)


        # main branch:






    def bottleneck_dilated(self, x):
        test =1

    def bottleneck_asymmetric(self, x):
        test =1
