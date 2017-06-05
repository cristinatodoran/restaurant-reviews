import tensorflow as tf
import numpy as np
import utils
import datetime,time
import os
from tensorflow.contrib import learn

# Configuration
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "reviews\\positive_reviews.dat", "Positive Examples")
tf.flags.DEFINE_string("negative_data_file", "reviews\\negative_reviews.dat", "Negative Examples.")

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def get_training_data(positive_data_file, negative_data_file):

    positive_data = list(open(positive_data_file, "r").readlines())
    positive_data = [s.strip() for s in positive_data]
    negative_data = list(open(negative_data_file, "r").readlines())
    negative_data = [s.strip() for s in negative_data]
    positive_labels = [[0, 1] for _ in positive_data]
    negative_labels = [[1, 0] for _ in negative_data]

    x_text = positive_data + negative_data
    x_text = [utils.filter_lower(sent) for sent in x_text]

    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


"""
A CNN for text classification.
Analyse customer's reviews (negative, postive)
"""
class ReviewsCNN(object):

    def __init__(self, MAX_SEQ_LEN, num_classes, vocab_size,
                 embedding_size, sliding_window_sizes, num_filters, l2_reg_lambda=0.0):

        self.train_X = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name="input_x")
        self.train_Y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # learns to map word vectors into a lower dimensional vector space
        # where distances between words correspond to how related they are
        # lookup table that we learn from data
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.train_X)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #  Create  convolution layer
        #  maxpool layer for each filter size
        layer_2_out = []
        for i, n_gram in enumerate(sliding_window_sizes):
            with tf.name_scope("conv-maxpool-%s" % n_gram):
                filter_shape = [n_gram, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                convolution_layer = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(convolution_layer, b), name="relu")

                """"
                feature vector, where the last dimension corresponds to the features
                """
                max_pool_layer = tf.nn.max_pool(
                    h,
                    ksize=[1, MAX_SEQ_LEN - n_gram + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                layer_2_out.append(max_pool_layer)

        """
        each convolution produces tensors of different shapes we need to iterate through them,
        create a layer for each of them, and then merge the results into one big feature vector
        """
        total_filters = num_filters * len(sliding_window_sizes)
        self.h_pool = tf.concat(layer_2_out, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters])

        """
         dropout regularization, which will randomly disable a fraction of neurons in the layer (set to 50% here)
         to ensure that that model does not overfit.
         preventing neurons from co-adapting and forcing them to learn individually useful features.
        """
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[total_filters, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.train_Y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.train_Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")







print("Preparing Training Data")
X, Y = get_training_data(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
print("Building Dictionary")
max_document_length = max([len(x.split(" ")) for x in X])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(X)))


# Randomly shuffle data
print("Shuffle Data Randomly")
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = x[shuffle_indices]
y_shuffled = Y[shuffle_indices]


dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(Y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]




with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ReviewsCNN(
            MAX_SEQ_LEN=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            sliding_window_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        vocab_processor.save(os.path.join(out_dir, "vocab"))


        sess.run(tf.global_variables_initializer())

        def step(x_batch, y_batch):

            feed_dict = {
              cnn.train_X: x_batch,
              cnn.train_Y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):

            feed_dict = {
              cnn.train_X: x_batch,
              cnn.train_Y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        batches = utils.get_next_batch(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
