import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

class Model:
	def __init__(self,params):

		n_hidden = params['n_hidden']
		n_input = params['n_input']
		
		# Input
		x_shape = [
				params['batch_size'],
				params['n_steps'],
				params['n_input']]
		self.x = tf.placeholder("float",x_shape)
		self._x = tf.transpose(self.x, [1, 0, 2])  

		# Reshape to prepare input for hidden activation
		self._x = tf.reshape(self._x, [-1, params['n_input']])
	
		# Target
		self.y = tf.placeholder("float", [None, params['n_input']])	
		
		# Initial State for each of the RNN Hidden layers
		self.initial = tf.placeholder("float", [None, 2*n_hidden])

		# Weights
		self.weights = {
			'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
			'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
		}

		# Biases
		self.biases = {
			'hidden': tf.Variable(tf.random_normal([n_hidden])),
			'out': tf.Variable(tf.random_normal([n_input]))
		}

		# Convinience
		_x = self._x
		_y = self.y
		weights = self.weights
		biases = self.biases

		# Linear Activation (of the hidden layers..?)
		hidden = tf.matmul(_x, weights['hidden']) + biases['hidden']

		# Split data because rnn cell needs a list of inputs 
		hidden = tf.split(0, params['n_steps'], hidden)

		# Define a lstm cell with tensorflow
		lstm_cell = rnn_cell.BasicLSTMCell(params['n_hidden'], forget_bias=1.0)

		# Get lstm cell output
		outputs, states = rnn.rnn(lstm_cell, hidden, initial_state=self.initial)

		# Linear activation
		# Get inner loop last output
		last = tf.matmul(outputs[-1], weights['out']) + biases['out']
		
		# Prediction
		self.pred = tf.nn.sigmoid(last)

		# Cost and Optimizer
		scel = tf.nn.sigmoid_cross_entropy_with_logits
		ao = tf.train.AdamOptimizer
		self.cost = tf.reduce_mean(scel(last,self.y))
		self.opt = ao(learning_rate=params['learning_rate'])
		self.opt = self.opt.minimize(self.cost)

		# Cosine Similiarity
		self.cos_sim = self.cosine_similiarity(self.y,self.pred)

		# Accuracy
		self.acc = tf.reduce_mean(tf.cast(self.cos_sim, tf.float32))
	
	def cosine_similiarity(self,_x,_y):
		dot = tf.reduce_sum(tf.mul(_x,_y))
		x_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(_x))))
		y_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(_y))))
		norms = tf.mul(x_norm,y_norm)
		return tf.div(dot, norms)


