import numpy as np
import tensorflow as tf

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

    def create_feed_dict(self, imgs_batch, labels_batch=None, keep_prob=1):
        """
        - DOES: returns a feed_dict mapping the placeholders to the actual
        input data (this is how we run the network on specific data).
        """

        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.keep_prob_ph] = keep_prob
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
