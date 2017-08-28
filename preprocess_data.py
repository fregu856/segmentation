import cv2
import cPickle
import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
import random

project_dir = "/home/fregu856/segmentation/"
cityscapes_dir = "/home/fregu856/data/cityscapes/"

# project_dir = "/root/segmentation/"
# cityscapes_dir = "/root/cityscapes/"

# (this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      1 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      1 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      1 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      1 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      1 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        1 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        1 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        1 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      1 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      1 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      1 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        1 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      1 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        1 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        1 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        1 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        1 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       1 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       1 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       1 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       1 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       1 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       1 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      1 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      1 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       1 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       1 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       1 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

id_to_class = {label.id: label.name for label in labels}

id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

original_img_height = 1024
original_img_width = 2048
new_img_height = 256
new_img_width = 512
no_of_classes = 2

train_imgs_dir = cityscapes_dir + "leftImg8bit/train/"
val_imgs_dir = cityscapes_dir + "leftImg8bit/val/"

train_gt_dir = cityscapes_dir + "gtFine/train/"
val_gt_dir = cityscapes_dir + "gtFine/val/"

# train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
#             "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
#             "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
#             "bremen/", "bochum/", "aachen/"]
train_dirs = ["jena/"]
#val_dirs = ["frankfurt/", "munster/", "lindau/"]
val_dirs = ["frankfurt/"]

pretrain_train_img_paths = []
pretrain_train_labels = []

train_img_paths = []
train_trainId_label_paths = []
for dir_step, dir in enumerate(train_dirs):
    img_dir = train_imgs_dir + dir

    file_names = os.listdir(img_dir)

    for step, file_name in enumerate(file_names):
        print "train dir %d, step %d" % (dir_step, step)

        img_path = img_dir + file_name
        img = cv2.imread(img_path, -1)
        img_id = file_name.split("_left")[0]

        img_small = cv2.resize(img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)
        img_small_path = project_dir + "data/" + img_id + ".png"
        cv2.imwrite(img_small_path, img_small)
        train_img_paths.append(img_small_path)

        gt_img_path = train_gt_dir + dir + img_id + "_gtFine_labelIds.png"
        gt_img = cv2.imread(gt_img_path, -1)
        gt_img_small = cv2.resize(gt_img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)

        id_label = gt_img_small
        trainId_label = id_to_trainId_map_func(id_label)

        trainId_label_path = project_dir + "data/" + img_id + "_trainId_label.png"
        cv2.imwrite(trainId_label_path, trainId_label)
        train_trainId_label_paths.append(trainId_label_path)



        img_small_flipped = cv2.flip(img_small, 1)
        img_small_flipped_path = project_dir + "data/" + img_id + "_flipped.png"
        cv2.imwrite(img_small_flipped_path, img_small_flipped)
        train_img_paths.append(img_small_flipped_path)

        gt_img_small_flipped = cv2.flip(gt_img_small, 1)

        id_label_flipped = gt_img_small_flipped
        trainId_label_flipped = id_to_trainId_map_func(id_label_flipped)

        trainId_label_flipped_path = project_dir + "data/" + img_id + "_trainId_label_flipped.png"
        cv2.imwrite(trainId_label_flipped_path, trainId_label_flipped)
        train_trainId_label_paths.append(trainId_label_flipped_path)



        gt_trainId = id_to_trainId_map_func(gt_img)
        for col in range(8):
            for row in range(8):
                img_crop = img[row*128:(row + 1)*128, col*256:(col + 1)*256]
                gt_crop = gt_trainId[row*256:(row + 1)*256, col*256:(col + 1)*256]

                for trainId in range(no_of_classes):
                    trainId_mask = np.equal(gt_crop, trainId)
                    trainId_count = np.sum(trainId_mask)
                    trainId_prop = float(trainId_count)/float(128*64)
                    if trainId_prop > 0.99:
                        img_crop_path = project_dir + "data/" + img_id + "_" + str(row) + "_" + str(col) + ".png"
                        cv2.imwrite(img_crop_path, img_crop)
                        pretrain_train_img_paths.append(img_crop_path)
                        pretrain_train_labels.append(trainId)
                        break

cPickle.dump(pretrain_train_img_paths,
            open(project_dir + "data/pretrain_train_img_paths.pkl", "w"))
cPickle.dump(pretrain_train_labels,
            open(project_dir + "data/pretrain_train_labels", "w"))
print len(pretrain_train_labels)



train_data = zip(train_img_paths, train_trainId_label_paths)
random.shuffle(train_data)
random.shuffle(train_data)
random.shuffle(train_data)
random.shuffle(train_data)
train_img_paths, train_trainId_label_paths = zip(*train_data)

cPickle.dump(train_trainId_label_paths,
            open(project_dir + "data/train_trainId_label_paths.pkl", "w"))
cPickle.dump(train_img_paths,
            open(project_dir + "data/train_img_paths.pkl", "w"))

train_trainId_label_paths = cPickle.load(open(project_dir + "data/train_trainId_label_paths.pkl"))
train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))


no_of_train_imgs = len(train_img_paths)
mean_img = np.zeros((new_img_height, new_img_width, 3))
for img_path in train_img_paths:
    img = cv2.imread(img_path, -1)
    mean_img += img

mean_img = mean_img/float(no_of_train_imgs)

cPickle.dump(mean_img, open(project_dir + "data/mean_img.pkl", "w"))










# trainId_to_count = {}
# for trainId in range(no_of_classes):
#     trainId_to_count[trainId] = 0
#
# for step, trainId_label_path in enumerate(train_trainId_label_paths):
#     print step
#
#     trainId_label = cv2.imread(trainId_label_path, -1)
#
#     for trainId in range(no_of_classes):
#         trainId_mask = np.equal(trainId_label, trainId)
#         label_trainId_count = np.sum(trainId_mask)
#
#         trainId_to_count[trainId] += label_trainId_count
#
# class_weights = []
# total_count = sum(trainId_to_count.values())
# for trainId, count in trainId_to_count.items():
#     trainId_prob = float(count)/float(total_count)
#     trainId_weight = 1/np.log(1.02 + trainId_prob)
#     class_weights.append(trainId_weight)
#
# print class_weights
#
# cPickle.dump(class_weights, open(project_dir + "data/class_weights.pkl", "w"))

no_of_pixels_in_img = new_img_width*new_img_height

trainId_to_count = {}
trainId_to_total_count = {}
for trainId in range(no_of_classes):
    trainId_to_count[trainId] = 0
    trainId_to_total_count[trainId] = 0

for step, trainId_label_path in enumerate(train_trainId_label_paths):
    print step

    trainId_label = cv2.imread(trainId_label_path, -1)

    for trainId in range(no_of_classes):
        trainId_mask = np.equal(trainId_label, trainId)
        trainId_count = np.sum(trainId_mask)

        if trainId_count > 0:
            trainId_to_count[trainId] += trainId_count
            trainId_to_total_count[trainId] += no_of_pixels_in_img

trainId_to_freq = {}
for trainId in range(no_of_classes):
    trainId_to_freq[trainId] = float(trainId_to_count[trainId])/float(trainId_to_total_count[trainId])

median_freq = np.median(trainId_to_freq.values())

class_weights = []
for trainId in range(no_of_classes):
    class_weights.append(float(median_freq)/float(trainId_to_freq[trainId]))

print class_weights

cPickle.dump(class_weights, open(project_dir + "data/class_weights.pkl", "w"))







pretrain_val_img_paths = []
pretrain_val_labels = []

val_img_paths = []
val_trainId_label_paths = []
for dir_step, dir in enumerate(val_dirs):
    img_dir = val_imgs_dir + dir

    file_names = os.listdir(img_dir)

    for step, file_name in enumerate(file_names):
        print "val dir %d, step %d" % (dir_step, step)

        img_path = img_dir + file_name
        img = cv2.imread(img_path, -1)
        img_id = file_name.split("_left")[0]

        img_small = cv2.resize(img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)
        img_small_path = project_dir + "data/" + img_id + ".png"
        cv2.imwrite(img_small_path, img_small)
        val_img_paths.append(img_small_path)

        gt_img_path = val_gt_dir + dir + img_id + "_gtFine_labelIds.png"
        gt_img = cv2.imread(gt_img_path, -1)
        gt_img_small = cv2.resize(gt_img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)

        id_label = gt_img_small
        trainId_label = id_to_trainId_map_func(id_label)

        trainId_label_path = project_dir + "data/" + img_id + "_trainId_label.png"
        cv2.imwrite(trainId_label_path, trainId_label)
        val_trainId_label_paths.append(trainId_label_path)



        gt_trainId = id_to_trainId_map_func(gt_img)
        for col in range(8):
            for row in range(8):
                img_crop = img[row*128:(row + 1)*128, col*256:(col + 1)*256]
                gt_crop = gt_trainId[row*256:(row + 1)*256, col*256:(col + 1)*256]

                for trainId in range(no_of_classes):
                    trainId_mask = np.equal(gt_crop, trainId)
                    trainId_count = np.sum(trainId_mask)
                    trainId_prop = float(trainId_count)/float(128*64)
                    if trainId_prop > 0.99:
                        img_crop_path = project_dir + "data/" + img_id + "_" + str(row) + "_" + str(col) + ".png"
                        cv2.imwrite(img_crop_path, img_crop)
                        pretrain_val_img_paths.append(img_crop_path)
                        pretrain_val_labels.append(trainId)
                        break

cPickle.dump(pretrain_val_img_paths,
            open(project_dir + "data/pretrain_val_img_paths.pkl", "w"))
cPickle.dump(pretrain_val_labels,
            open(project_dir + "data/pretrain_val_labels", "w"))
print len(pretrain_val_labels)




cPickle.dump(val_trainId_label_paths,
            open(project_dir + "data/val_trainId_label_paths.pkl", "w"))
cPickle.dump(val_img_paths,
            open(project_dir + "data/val_img_paths.pkl", "w"))

val_trainId_label_paths = cPickle.load(open(project_dir + "data/val_trainId_label_paths.pkl"))
val_img_paths = cPickle.load(open(project_dir + "data/val_img_paths.pkl"))
