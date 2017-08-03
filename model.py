import numpy as np
import tensorflow as tf
import os

from utilities import PReLU, spatial_dropout, max_unpool

class ENet_model(object):
    """
    - DOES:
    """

    def __init__(self, model_id):
        """
        - DOES:
        """

        self.model_id = model_id

        self.lr = 0.001

        self.logs_dir = "/home/fregu856/segmentation/training_logs"
        self.no_of_classes = 16
        self.class_weights = [0, 0.5, 0.4] # TODO!
        self.initial_lr = 5e-4 # TODO!
        self.decay_steps =  1000 # TODO!
        self.lr_decay_rate = 1e-1 # TODO!
        self.img_height = 512
        self.img_width = 1024

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
        """

        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

    def add_placeholders(self):
        """
        - DOES:
        """

        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[None, self.img_height, self.img_width, 3], # ([batch_size, img_heigth, img_width, 3])
                    name="imgs_ph")
        self.onehot_labels_ph = tf.placeholder(tf.uint8,
                    shape=[None, self.img_height, self.img_width, self.no_of_classes], # ([batch_size, img_heigth, img_width, no_of_classes])
                    name="onehot_labels_ph")
        self.training_ph = tf.placeholder(tf.bool, name="training_ph")

    def create_feed_dict(self, imgs_batch, onehot_labels_batch=None, keep_prob=1, training=True):
        """
        - DOES: returns a feed_dict mapping the placeholders to the actual
        input data (this is how we run the network on specific data).
        """

        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.training_ph] = training
        if onehot_labels_batch is not None:
            # only add the labels data if it's specified (during inference, we
            # won't have any labels):
            feed_dict[self.onehot_labels_ph] = onehot_labels_batch

    def add_logits(self):
        """
        - DOES:
        """

        # encoder:
        initial = self.initial_block(x=self.imgs_ph, scope="inital")

        bottleneck_1_0, pooling_indices_1 = self.encoder_bottleneck_regular(x=initial,
                    output_depth=64, drop_prob=0.01, scope="bottleneck_1_0", downsampling=True)
        bottleneck_1_1 = self.encoder_bottleneck_regular(x=bottleneck_1_0,
                    output_depth=64, drop_prob=0.01, scope="bottleneck_1_1")
        bottleneck_1_2 = self.encoder_bottleneck_regular(x=bottleneck_1_1,
                    output_depth=64, drop_prob=0.01, scope="bottleneck_1_2")
        bottleneck_1_3 = self.encoder_bottleneck_regular(x=bottleneck_1_2,
                    output_depth=64, drop_prob=0.01, scope="bottleneck_1_3")
        bottleneck_1_4 = self.encoder_bottleneck_regular(x=bottleneck_1_3,
                    output_depth=64, drop_prob=0.01, scope="bottleneck_1_4")

        bottleneck_2_0, pooling_indices_2 = self.encoder_bottleneck_regular(x=bottleneck_1_4,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_0", downsampling=True)
        bottleneck_2_1 = self.encoder_bottleneck_regular(x=bottleneck_2_0,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_1")
        bottleneck_2_2 = self.encoder_bottleneck_dilated(x=bottleneck_2_1,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_2", dilation_rate=2)
        bottleneck_2_3 = self.encoder_bottleneck_asymmetric(x=bottleneck_2_2,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_3")
        bottleneck_2_4 = self.encoder_bottleneck_dilated(x=bottleneck_2_3,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_4", dilation_rate=4)
        bottleneck_2_5 = self.encoder_bottleneck_regular(x=bottleneck_2_4,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_5")
        bottleneck_2_6 = self.encoder_bottleneck_dilated(x=bottleneck_2_5,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_6", dilation_rate=8)
        bottleneck_2_7 = self.encoder_bottleneck_asymmetric(x=bottleneck_2_6,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_7")
        bottleneck_2_8 = self.encoder_bottleneck_dilated(x=bottleneck_2_7,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_2_8", dilation_rate=16)

        bottleneck_3_1 = self.encoder_bottleneck_regular(x=bottleneck_2_8,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_1")
        bottleneck_3_2 = self.encoder_bottleneck_dilated(x=bottleneck_3_1,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_2", dilation_rate=2)
        bottleneck_3_3 = self.encoder_bottleneck_asymmetric(x=bottleneck_3_2,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_3")
        bottleneck_3_4 = self.encoder_bottleneck_dilated(x=bottleneck_3_3,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_4", dilation_rate=4)
        bottleneck_3_5 = self.encoder_bottleneck_regular(x=bottleneck_3_4,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_5")
        bottleneck_3_6 = self.encoder_bottleneck_dilated(x=bottleneck_3_5,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_6", dilation_rate=8)
        bottleneck_3_7 = self.encoder_bottleneck_asymmetric(x=bottleneck_3_6,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_7")
        bottleneck_3_8 = self.encoder_bottleneck_dilated(x=bottleneck_3_7,
                    output_depth=128, drop_prob=0.1, scope="bottleneck_3_8", dilation_rate=16)

        # decoder:
        bottleneck_4_0 = self.decoder__bottleneck(x=bottleneck_3_8,
                    output_depth=64, scope="bottleneck_4_0",
                    upsampling=True, pooling_indices=pooling_indices_2)
        bottleneck_4_1 = self.decoder__bottleneck(x=bottleneck_4_0,
                    output_depth=64, scope="bottleneck_4_1")
        bottleneck_4_2 = self.decoder__bottleneck(x=bottleneck_4_1,
                    output_depth=64, scope="bottleneck_4_2")

        bottleneck_5_0 = self.decoder__bottleneck(x=bottleneck_4_2,
                    output_depth=16, scope="bottleneck_5_0",
                    upsampling=True, pooling_indices=pooling_indices_1)
        bottleneck_5_1 = self.decoder__bottleneck(x=bottleneck_5_0,
                    output_depth=16, scope="bottleneck_5_1")

        # fullconv:
        fullconv = tf.contrib.slim.conv2d_transpose(bottleneck_5_1, self.no_of_classes,
                    [2, 2], stride=2, scope="fullconv", padding="SAME", activation_fn=None)

        self.logits = fullconv

    def add_loss_op(self):
        """
        - DOES: computes the weighted CE loss for the batch.
        """

        weights = self.onehot_labels_ph*self.class_weights
        weights = tf.reduce_sum(weights, 3)
        # compute the weighted CE loss for each pixel in the batch:
        loss_per_pixel = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels_ph,
                    logits=self.logits, weights=weights)
        # average the loss over all pixels to get the batch loss:
        self.loss = tf.reduce_mean(loss_per_pixel)

    def add_train_op(self):
        """
        - DOES: creates a training operator for minimization of the loss.
        """

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate=self.initial_lr,
                    global_step=global_step, decay_steps=self.decay_steps,
                    decay_rate=self.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step) # (global_step will now automatically be incremented)

    def initial_block(self, x, scope):
        # convolution branch:
        W_conv = tf.get_variable(scope + "/W",
                    shape=[3, 3, 3, 13], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        b_conv = tf.get_variable(scope + "/b", shape=[13], # ([out_depth]], one bias weight per out depth layer),
                    initializer=tf.constant_initializer(0))
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                    padding="SAME") + b_conv

        # max pooling branch:
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding="VALID")

        # concatenate the branches:
        concat = tf.concat([conv_branch, pool_branch], axis=3) # (3: the depth axis)

        # apply batch normalization and PReLU:
        output = tf.contrib.slim.batch_norm(concat,
                    is_training=self.training_ph)
        output = PReLU(output, scope=scope)

        return output

    def encoder_bottleneck_regular(self, x, output_depth, drop_prob, scope, proj_ratio=4, downsampling=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        if downsampling:
            W_conv = tf.get_variable(scope + "/W_proj",
                        shape=[2, 2, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 2, 2, 1],
                        padding="VALID") # NOTE! no bias terms
        else:
            W_proj = tf.get_variable(scope + "/W_proj",
                        shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # conv:
        W_conv = tf.get_variable(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        b_conv = tf.get_variable(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                    initializer=tf.constant_initializer(0))
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

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
        main_branch = x

        if downsampling:
            # # max pooling with argmax (for use in upsampling in the decoder):
            main_branch, pooling_indices = tf.nn.max_pool_with_argmax(main_branch,
                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # (everytime we downsample, we also increase the feature block depth)

            # # pad with zeros so that the feature block depth matches:
            depth_to_pad = output_depth - input_depth
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            # (paddings is an integer tensor of shape [4, 2] where 4 is the rank
            # of main_branch. For each dimension D (D = 0, 1, 2, 3) of main_branch,
            # paddings[D, 0] is the no of values to add before the contents of main_branch
            # in that dimension, and paddings[D, 0] is the no of values to add after
            # the contents of main_branch in that dimension.)
            main_branch = tf.pad(main_branch, paddings=paddings, mode="CONSTANT")

        # add the branches:
        merged = conv_branch + main_branch
        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        if downsampling:
            return output, pooling_indices
        else:
            return output

    def encoder_bottleneck_dilated(self, x, output_depth, drop_prob, scope, dilation_rate, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = tf.get_variable(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # dilated conv:
        W_conv = tf.get_variable(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        b_conv = tf.get_variable(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                    initializer=tf.constant_initializer(0))
        conv_branch = tf.nn.atrous_conv2d(conv_branch, W_conv, rate=dilation_rate,
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

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
        main_branch = x

        # add the branches:
        merged = conv_branch + main_branch
        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def encoder_bottleneck_asymmetric(self, x, output_depth, drop_prob, scope, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = tf.get_variable(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # asymmetric conv:
        # # # asymmetric conv 1:
        W_conv1 = tf.get_variable(scope + "/W_conv1",
                    shape=[5, 1, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        conv_branch = tf.nn.conv2d(conv_branch, W_conv1, strides=[1, 1, 1, 1],
                    padding="SAME") # NOTE! no bias terms
        # # # asymmetric conv 2:
        W_conv2 = tf.get_variable(scope + "/W_conv2",
                    shape=[1, 5, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        b_conv2 = tf.get_variable(scope + "/b_conv2", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                    initializer=tf.constant_initializer(0))
        conv_branch = tf.nn.conv2d(conv_branch, W_conv2, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv2
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

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
        main_branch = x

        # add the branches:
        merged = conv_branch + main_branch
        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def decoder_bottleneck(self, x, output_depth, scope, proj_ratio=4, upsampling=False, pooling_indices=None):
        # (decoder uses ReLU instead of PReLU)

        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # main branch:
        main_branch = x

        if upsampling:
            # # 1x1 projection (to decrease depth to the same value as before downsampling):
            W_upsample = tf.get_variable(scope + "/W_upsample",
                        shape=[1, 1, input_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            main_branch = tf.nn.conv2d(main_branch, W_upsample, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
            # # # batch norm:
            main_branch = tf.contrib.slim.batch_norm(main_branch, is_training=self.training_ph)
            # NOTE! no ReLU here

            # # max unpooling:
            main_branch = max_unpool(main_branch, pooling_indices)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = tf.get_variable(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = tf.nn.relu(conv_branch)

        # # conv:
        if upsampling:
            # deconvolution:
            W_conv = tf.get_variable(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            b_conv = tf.get_variable(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                        initializer=tf.constant_initializer(0))
            main_branch_shape = main_branch.get_shape().as_list()
            output_shape = tf.convert_to_tensor([main_branch_shape[0],
                        main_branch_shape[1], main_branch_shape[2], internal_depth])
            conv_branch = tf.nn.conv2d_transpose(conv_branch, W_conv, output_shape=output_shape,
                        strides=[1, 2, 2, 1], padding="SAME") + b_conv
        else:
            W_conv = tf.get_variable(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer())
            b_conv = tf.get_variable(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                        initializer=tf.constant_initializer(0))
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                        padding="SAME") + b_conv
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        conv_branch = tf.nn.relu(conv_branch)

        # # 1x1 expansion:
        W_exp = tf.get_variable(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer())
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch, is_training=self.training_ph)
        # NOTE! no ReLU here

        # NOTE! no regularizer

        # add the branches:
        merged = conv_branch + main_branch
        # apply ReLU:
        output = tf.nn.relu(merged)

        return output
