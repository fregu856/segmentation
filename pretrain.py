import numpy as np
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random

from model import ENet_model

#project_dir = "/home/fregu856/segmentation/"
project_dir = "/root/segmentation/"

data_dir = project_dir + "data/"

model_id = "pretrain_1" # (change this to not overwrite all log data when you train the model)
batch_size = 64
img_height = 128
img_width = 256

model = ENet_model(model_id, img_height=img_height, img_width=img_width, batch_size=batch_size)

train_mean_channels = cPickle.load(open("data/pretrain_mean_channels.pkl"))

# load the training data from disk:
train_img_paths = cPickle.load(open(data_dir + "pretrain_train_img_paths.pkl"))
train_labels = cPickle.load(open(data_dir + "pretrain_train_labels.pkl"))
train_data = zip(train_img_paths, train_labels)

# compute the number of batches needed to iterate through the training data:
no_of_train_imgs = len(train_img_paths)
no_of_batches = int(no_of_train_imgs/batch_size)

# load the validation data from disk:
val_img_paths = cPickle.load(open(data_dir + "pretrain_val_img_paths.pkl"))
val_labels = cPickle.load(open(data_dir + "pretrain_val_labels.pkl"))
val_data = zip(val_img_paths, val_labels)

# compute the number of batches needed to iterate through the val data:
no_of_val_imgs = len(val_img_paths)
no_of_val_batches = int(no_of_val_imgs/batch_size)

def evaluate_on_val(batch_size, sess):
    """
    - DOES:
    """

    print "evaluation on val:"

    random.shuffle(val_data)
    val_img_paths, val_labels = zip(*val_data)

    val_batch_losses = []
    batch_pointer = 0
    for step in range(no_of_val_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_labels = []
        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(val_img_paths[batch_pointer + i], -1)
            img = img - train_mean_channels
            batch_imgs[i] = img

            batch_labels.append(val_labels[(batch_pointer + i)])
        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(batch_imgs, early_drop_prob=0.0,
                    late_drop_prob=0.0, training=False, pretrain_labels_batch=batch_labels)

        batch_loss, logits = sess.run([model.pretrain_loss, model.pretrain_logits],
                    feed_dict=batch_feed_dict)
        val_batch_losses.append(batch_loss)
        print "epoch: %d/%d, val step: %d/%d, val batch loss: %g" % (epoch+1, no_of_epochs, step+1, no_of_val_batches, batch_loss)

        predictions = np.argmax(logits, axis=1)
        no_of_roads = np.count_nonzero(predictions == 0)
        no_of_nonroads = np.count_nonzero(predictions == 1)
        print ("predictions on val: roads: %d/%d, nonroads: %d/%d" % (no_of_roads,
                    batch_size, no_of_nonroads, batch_size))

    val_loss = np.mean(val_batch_losses)
    return val_loss

def train_data_iterator(batch_size, session):
    """
    - DOES:
    """

    random.shuffle(train_data)
    train_img_paths, train_labels = zip(*train_data)

    batch_pointer = 0
    for step in range(no_of_batches):
        # get and yield the next batch_size imgs and labels from the training data:
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_labels = []
        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(train_img_paths[batch_pointer + i], -1)
            img = img - train_mean_channels
            batch_imgs[i] = img

            batch_labels.append(train_labels[(batch_pointer + i)])
        batch_pointer += batch_size

        yield (batch_imgs, batch_labels)

no_of_epochs = 120

# create a saver for saving all model variables/parameters:
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

# initialize all log data containers:
train_loss_per_epoch = []
val_loss_per_epoch = []

# initialize a list containing the 5 best val losses (is used to tell when it
# makes sense to save a model checkpoint):
best_epoch_losses = [1000, 1000, 1000, 1000, 1000]

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    #saver.restore(sess, "/home/fregu856/2D_detection/training_logs/model_1/checkpoints/model_1_epoch_1.ckpt")

    for epoch in range(no_of_epochs):
        print "###########################"
        print "######## NEW EPOCH ########"
        print "###########################"
        print "epoch: %d/%d" % (epoch+1, no_of_epochs)

        # run an epoch and get all batch losses:
        batch_losses = []
        for step, (imgs, labels) in enumerate(train_data_iterator(batch_size, sess)):
            # create a feed dict containing the batch data:
            batch_feed_dict = model.create_feed_dict(imgs, early_drop_prob=0.01,
                        late_drop_prob=0.1, training=True, pretrain_labels_batch=labels)

            # compute the batch loss and compute & apply all gradients w.r.t to
            # the batch loss (without model.pretrain_train_op in the call, the network
            # would NOT train, we would only compute the batch loss):
            batch_loss, _ = sess.run([model.pretrain_loss, model.pretrain_train_op],
                        feed_dict=batch_feed_dict)
            batch_losses.append(batch_loss)

            print "step: %d/%d, training batch loss: %g" % (step+1, no_of_batches, batch_loss)

        # compute the train epoch loss:
        train_epoch_loss = np.mean(batch_losses)
        # save the train epoch loss:
        train_loss_per_epoch.append(train_epoch_loss)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss_per_epoch, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))
        print "training loss: %g" % train_epoch_loss

        # run the model on the validation data:
        val_loss = evaluate_on_val(batch_size, sess)
        # save the val epoch loss:
        val_loss_per_epoch.append(val_loss)
        # save the val epoch losses to disk:
        cPickle.dump(val_loss_per_epoch, open("%sval_loss_per_epoch.pkl"\
                    % model.model_dir, "w"))
        print "validaion loss: %g" % val_loss

        if val_loss < max(best_epoch_losses): # (if top 5 performance on val:)
            # save the model weights to disk:
            checkpoint_path = (model.checkpoints_dir + "model_" +
                        model.model_id + "_epoch_" + str(epoch + 1) + ".ckpt")
            saver.save(sess, checkpoint_path)
            print "checkpoint saved in file: %s" % checkpoint_path

            # update the top 5 val losses:
            index = best_epoch_losses.index(max(best_epoch_losses))
            best_epoch_losses[index] = val_loss

        # plot the training loss vs epoch and save to disk:
        plt.figure(1)
        plt.plot(train_loss_per_epoch, "k^")
        plt.plot(train_loss_per_epoch, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("training loss per epoch")
        plt.savefig("%strain_loss_per_epoch.png" % model.model_dir)
        plt.close(1)

        # plot the val loss vs epoch and save to disk:
        plt.figure(1)
        plt.plot(val_loss_per_epoch, "k^")
        plt.plot(val_loss_per_epoch, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("validation loss per epoch")
        plt.savefig("%sval_loss_per_epoch.png" % model.model_dir)
        plt.close(1)
