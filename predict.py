import tensorflow as tf
import numpy as np
import os

import utils
from tensorflow.contrib import learn
import csv
import time

tf.flags.DEFINE_string("test_data", "reviews\\reviews_notlabeled.dat", "Data source for the positive data.")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

#can't be run on my GPU- not enugh rams :(
#can be run on CPU, but it takes 24 h for 500 epochs
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

def get_prediction_data(data_file):

    # Load data from files
    data_ = list(open(data_file, "r").readlines())
    data_ = [s.strip() for s in data_]

    # Split by words
    x_text = data_
    x_text = [utils.filter_lower(sent) for sent in x_text]

    return x_text




#get data for prediction
x_raw  = get_prediction_data(FLAGS.test_data)

# restore form checkpoint
# model already trained
path_vocab = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(path_vocab)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nPrediction using Convolutional Neural Networks\n")

print ("Restore from specific check point")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # generator of batches
        batches = utils.get_next_batch(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            print (all_predictions)

print ("total reviews for predictions:", len(x_test))
pos =0
neg =0
for x in range(len(all_predictions)):
    if all_predictions[x] == 1:
        pos +=1
    else:
        neg +=1
print ("Positive reviews:",pos)
print ("Negative reviews:",neg)
predictedLabels = np.column_stack((np.array(x_raw), all_predictions))
timestr = time.strftime("%Y%m%d-%H%M%S")
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction"+timestr+".csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictedLabels)