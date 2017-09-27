import numpy as np
import cPickle
import tensorflow as tf
import cv2
import os

from utilities import label_img_to_color

from model import ENet_model

project_dir = "/root/segmentation/"

data_dir = project_dir + "data/"

model_id = "sequence_run"

batch_size = 4
img_height = 512
img_width = 1024

model = ENet_model(model_id, img_height=img_height, img_width=img_width,
            batch_size=batch_size)

no_of_classes = model.no_of_classes

# load the mean color channels of the train imgs:
train_mean_channels = cPickle.load(open("data/mean_channels.pkl"))

# load the sequence data:
seq_frames_dir = "/root/data/cityscapes/leftImg8bit/demoVideo/stuttgart_02/"
seq_frame_paths = []
frame_names = sorted(os.listdir(seq_frames_dir))
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    frame_path = seq_frames_dir + frame_name
    seq_frame_paths.append(frame_path)

# compute the number of batches needed to iterate through the data:
no_of_frames = len(seq_frame_paths)
no_of_batches = int(no_of_frames/batch_size)

# define where to place the resulting images:
results_dir = model.project_dir + "results_on_seq/"

# create a saver for restoring variables/parameters:
saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    # restore the best trained model:
    saver.restore(sess, project_dir + "training_logs/best_model/model_1_epoch_23.ckpt")

    batch_pointer = 0
    for step in range(no_of_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        img_paths = []

        for i in range(batch_size):
            img_path = seq_frame_paths[batch_pointer + i]
            img_paths.append(img_path)

            # read the image:
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (img_width, img_height))
            img = img - train_mean_channels
            batch_imgs[i] = img

        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(imgs_batch=batch_imgs,
                    early_drop_prob=0.0, late_drop_prob=0.0)

        # run a forward pass and get the logits:
        logits = sess.run(model.logits, feed_dict=batch_feed_dict)

        print "step: %d/%d" % (step+1, no_of_batches)

        # save all predicted label images overlayed on the input frames to results_dir:
        predictions = np.argmax(logits, axis=3)
        for i in range(batch_size):
            pred_img = predictions[i]
            pred_img_color = label_img_to_color(pred_img)

            img = batch_imgs[i] + train_mean_channels

            img_file_name = img_paths[i].split("/")[-1]
            img_name = img_file_name.split(".png")[0]
            pred_path = results_dir + img_name + "_pred.png"

            overlayed_img = 0.3*img + 0.7*pred_img_color

            cv2.imwrite(pred_path, overlayed_img)

# create a video of all the resulting overlayed images:
fourcc = cv2.cv.CV_FOURCC("M", "J", "P", "G")
out = cv2.VideoWriter(results_dir + "cityscapes_stuttgart_02_pred.avi", fourcc,
            20.0, (img_width, img_height))

frame_names = sorted(os.listdir(results_dir))
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    if ".png" in frame_name:
        frame_path = results_dir + frame_name
        frame = cv2.imread(frame_path, -1)

        out.write(frame)
