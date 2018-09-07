import numpy as np
import tensorflow as tf
import pandas as pd
import time
from copy import deepcopy

# Disable chain indexing warning - Not highly advised unless you know what your doing.
pd.options.mode.chained_assignment = None

# Neurons - Looking to place them in tf.save later
input_layer = 3
hidden_layer = 4
output_layer = 1

# Splice data
dataset = pd.read_csv('Logic.csv', sep=',')
values = list(dataset.columns.values)

# Grab first two columns and add additional column to directly link operator to input data set
X = dataset[values[0:2]]
X['o'] = pd.Series([0 for i in range(len(X['x'])) ], index=X.index)
copier = deepcopy(X)
for i in range(len(values) - 3):
	copier['o'] = [(i + 1)] * len(copier['o'])
	for j in range(len(copier)):
		X.loc[len(X)] = copier.loc[j]

# Grab after 2 inputs, n many columns. Keeps this more dynamic for future operators using a 2 input system or even 1 input (find later)
y_data = dataset[values[2:]]
Y = pd.DataFrame(columns=['r'])

for index in iter(y_data.to_dict().values()):
	for j in range(len(index)):
		Y.loc[len(Y)] = index[j]

X = np.array(X)
Y = np.array(Y)

# Shuffle Data
indices = np.random.choice(len(X), len(X), replace=False)
X_input = X[indices]
y_output = Y[indices]

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

	# Setup tensorboard writer
	writer = tf.summary.FileWriter("./logs/and_logs", sess.graph)

	sess.run(init)

	t_start = time.clock()
	for i in range(100000):
		sess.run(train_step, feed_dict={X_data : X_input, y_target : y_output })
		if i % 1000 == 0:
			print("Epoch ", i)
			print("Hypothesis ", sess.run(hypothesis, feed_dict={X_data : X_input, y_target : y_output}))
			print("Weight 1 ", sess.run(weight_one))
			print("Bias 1 ", sess.run(bias_one))
			print("Weight 2 ", sess.run(weight_two))
			print("Bias 2 ", sess.run(bias_two))
			print("cost ", sess.run(cost, feed_dict={X_data : X_input, y_target : y_output}))
	t_end = time.clock()
	print("Elapsed time ", (t_end - t_start))
	
	# Save to output due to training being complete
	save_path = saver.save(sess, "./saves/logic.ckpt")

	print("Testing logic operators.")
	for i in range(len(X_input)):
		print("Input {} and {}, operator {} --> Actual: ".format(X_input[i][0], X_input[i][1], values[X_input[i][2] + 2]), y_output[i], "Predicted: ", np.rint(sess.run(hypothesis, feed_dict={X_data : [X_input[i]]})))
