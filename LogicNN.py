import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import time

parse = argparse.ArgumentParser()
parse.add_argument("-a", "--and", action="store_true", help="evaluate 'and' logic gate.")
parse.add_argument("-o", "--or", action="store_true", help="evaluate 'or' logic gate.")
parse.add_argument("-x", "--xor", action="store_true", help="evaluate 'xor' logic gate.")
args = vars(parse.parse_args())

train_logic = 0
if args["or"]:
	train_logic = 1
elif args["xor"]:
	train_logic= 2
print(train_logic)
# neurons
input_layer = 2
hidden_layer = 4
output_layer = 1

# Splice data
dataset = pd.read_csv('Logic.csv', sep=',')
values = list(dataset.columns.values)

X_input = np.array(dataset[values[0:2]]).tolist()

# change to operator column to train on
y_output = np.array(dataset[values[train_logic + 2]])
y_test = []
for i in y_output.tolist():
	y_test.append([i])
y_output = y_test[:]

# Define initializers, used later for training predictions
# initialize to float32 for tensorflows used tensor datatype to be compatible
X_data = tf.placeholder(tf.float32, shape=[None,2], name='x-inputdata')
y_target = tf.placeholder(tf.float32, shape=[None,1], name='y-targetdata')

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

	saver.restore(sess, "./saves/{}.ckpt".format(values[train_logic + 2]))

	print("Testing {} operator.".format(values[train_logic + 2]))
	for i in range(len(y_output)):
		print("Actual: ", y_output[i], "Predicted: ", np.rint(sess.run(hypothesis, feed_dict={X_data : [X_input[i]]})))
