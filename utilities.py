import tensorflow as tf
import cv2
import numpy as np

def PReLU(x, scope):
    """
    - DOES:

    - INPUT:

    - OUTPUT:
    """

    # (based on the implementation by kwotsin)

    # (PReLU(x) = x if x > 0, alpha*x otherwise)

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha*(x - abs(x))*0.5

    return output

def spatial_dropout(x, drop_prob):
    """
    - DOES:

    - INPUT:

    - OUTPUT:
    """

    # (based on the implementation by kwotsin)

    # x is a tensor of shape [batch_size, fb_height, fb_width, fb_depth]
    # where fb stands for "feature block"

    keep_prob = 1.0 - drop_prob
    input_shape = x.get_shape().as_list()

    batch_size = input_shape[0]
    fb_depth = input_shape[3]

    # drop each feature block layer with probability drop_prob:
    noise_shape = tf.constant(value=[batch_size, 1, 1, fb_depth])
    x_drop = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)

    output = x_drop

    return output

# def max_unpool(x, pooling_indices, output_shape, kernel_shape=[1, 2, 2, 1]):
#     """
#     - DOES:
#
#     - INPUT:
#
#     - OUTPUT:
#     """
#
#     # TODO! understand this code properly and comment!
#
#     # (based on the implementation by kwotsin)
#
#     input_shape = x.get_shape().as_list()
#     batch_size, fb_height, fb_width, fb_depth = (input_shape[0], input_shape[1],
#                 input_shape[2], input_shape[3])
#
#     # output_shape = (batch_size, fb_height*kernel_shape[1],
#     #             fb_width*kernel_shape[2], fb_depth)
#
#     ones_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
#     batch_shape = tf.convert_to_tensor([batch_size, 1, 1, 1])
#     batch_range = tf.reshape(tf.range(batch_size, dtype=tf.int32), shape=batch_shape)
#
#     b = tf.cast(ones_like_pooling_indices*batch_range, tf.int32)
#     y = tf.cast(pooling_indices//(output_shape[2]*output_shape[3]), tf.int32)
#     x = tf.cast((pooling_indices//output_shape[3]) % output_shape[2], tf.int32)
#     feature_range = tf.range(fb_depth, dtype=tf.int32)
#     f = tf.cast(ones_like_pooling_indices*feature_range, tf.int32)
#
#     # transpose indices & reshape update values to one dimension
#     input_size = tf.size(x)
#     indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, input_size]))
#     values = tf.reshape(x, [input_size])
#
#     output = tf.scatter_nd(indices, values, output_shape)
#
#     return output

def max_unpool(updates, mask, output_shape=None, k_size=[1, 2, 2, 1]):
    '''
    Unpooling function based on the implementation by Panaetius at https://github.com/tensorflow/tensorflow/issues/2169
    INPUTS:
    - inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
    - mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
    - k_size(list): a list of values representing the dimensions of the unpooling filter.
    - output_shape(list): a list of values to indicate what the final output shape should be after unpooling
    - scope(str): the string name to name your scope
    OUTPUTS:
    - ret(Tensor): the returned 4D tensor that has the shape of output_shape.
    '''
    mask = tf.cast(mask, tf.int32)
    input_shape = tf.shape(updates, out_type=tf.int32)

    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_mask * feature_range

    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(updates)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(updates, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color
