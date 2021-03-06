import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import time

# Disable chain indexing warning - Not highly advised unless you know what your doing.
pd.options.mode.chained_assignment = None

parse = argparse.ArgumentParser()
parse.add_argument("-a", "--and", action="store_true", help="evaluate 'and' logic gate.")
parse.add_argument("-o", "--or", action="store_true", help="evaluate 'or' logic gate.")
parse.add_argument("-x", "--xor", action="store_true", help="evaluate 'xor' logic gate.")
parse.add_argument("-n", "--not", action="store_true", help="evaluate 'not' logic gate, compared to x input.")
args = vars(parse.parse_args())

train_logic = 0
if args["or"]:
	train_logic = 1
elif args["xor"]:
	train_logic= 2
elif args["not"]:
	train_logic = 3

# neurons
input_layer = 3
hidden_layer = 4
output_layer = 1

# Splice data
dataset = pd.read_csv('Logic.csv', sep=',')
values = list(dataset.columns.values)

# Grab first two columns and add additional column to directly link operator to input data set
X = dataset[values[0:2]]
X['o'] = pd.Series([train_logic for i in range(len(X['x'])) ], index=X.index)
X_input = np.array(X)

# change to operator column to train on
y_output = np.array(dataset[values[train_logic + 2]])
y_test = []
for i in y_output.tolist():
	y_test.append([i])
y_output = y_test[:]

# Define initializers, used later for training predictions
# initialize to float32 for tensorflows used tensor datatype to be compatible
X_data = tf.placeholder(tf.float32, shape=[None,input_layer], name='x-inputdata')
y_target = tf.placeholder(tf.float32, shape=[None,output_layer], name='y-targetdata')

# Randomly distribute within the shape input_layer,hidden_layer --> -1 to 1
# https://www.tensorflow.org/api_docs/python/tf/random_uniform
weight_one = tf.Variable(tf.random_uniform([input_layer,hidden_layer], -1, 1), name = "Weight_One")
weight_two = tf.Variable(tf.random_uniform([hidden_layer,output_layer], -1, 1), name = "Weight_Two")

bias_one = tf.Variable(tf.zeros([hidden_layer]), name="Bias_One")
bias_two = tf.Variable(tf.zeros([output_layer]), name="Bias_Two")

with tf.name_scope("layer2") as scope:
	synapse0 = tf.sigmoid(tf.matmul(X_data, weight_one) + bias_one, name="Synapse0")

with tf.name_scope("layer3") as scope:
	hypothesis = tf.sigmoid(tf.matmul(synapse0, weight_two) + bias_two, name="Hypothesis")

with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(( (y_target * tf.log(hypothesis)) + ((1 - y_target) * tf.log(1.0 - hypothesis)) ) * -1, name="Cost")

# Rationale behind GDO can be found on iamtrask NN
with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

# Save output of training
saver = tf.train.Saver()	

with tf.Session() as sess:

	saver.restore(sess, "./saves/logic.ckpt")

	print("Testing {} operator.".format(values[train_logic + 2]))
	for i in range(len(X_input)):
		print("Input {} and {}, operator {} --> Actual: ".format(X_input[i][0], X_input[i][1], values[X_input[i][2] + 2]), y_output[i], "Predicted: ", np.rint(sess.run(hypothesis, feed_dict={X_data : [X_input[i]]})))
