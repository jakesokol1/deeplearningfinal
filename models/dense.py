from __future__ import absolute_import
from matplotlib import pyplot as plt
from .classifier import Classifier
from .preprocess import get_data
from .subject_graph import SubjectGraph

import os
import sys
import tensorflow as tf
import random
import numpy as np

class DenseDepressionClassifier(Classifier):

	def __init__(self, mode, name):

		self.model_path = './graph_models/'

		print('Fetching Data...')
		self.training_data, self.testing_data = get_data('depression')
		print('Done.')

		self.name = name
		print(self.name)

		if mode == 'train':
			self.model = DenseModel()
		elif mode == 'test':
			assert(False)
		else:
			assert(False)


	def train(self):

		print('Training...')
		indices = []

		"""Separate into inputs and labels"""
		train_inputs = []
		train_labels = []

		for i in range(0, len(self.training_data)):
			train_inputs.append(self.training_data[i].nodes)
			train_labels.append(self.training_data[i].label)

		train_inputs = tf.convert_to_tensor(train_inputs, dtype='float32')
		train_labels = tf.convert_to_tensor(train_labels, dtype='float32')

		"""Training"""

		for j in range(1,self.model.number_of_epochs+1):
			print("---------------------- EPOCH %d ----------------------" %j)

			if(j % 5 == 1):
				print("---------------Test results before EPOCH %d---------------" %j)
				test_inputs = []
				test_labels = []
				for i in range(0, len(self.testing_data)):
					test_inputs.append(self.testing_data[i].nodes)
					test_labels.append(self.testing_data[i].label)

				test_inputs = tf.convert_to_tensor(test_inputs, dtype='float32')
				test_labels = tf.convert_to_tensor(test_labels, dtype='int32')
				predictions = self.model.call(test_inputs)
				accuracy = self.model.accuracy(predictions, test_labels)
				print("Accuracy: ")
				print(accuracy)

			for i in range(0,len(self.training_data), self.model.batch_size):
				if((self.model.number_of_examples-i)>=self.model.batch_size):
					current_inputs = train_inputs[i:i+self.model.batch_size]
					current_labels = train_labels[i:i+self.model.batch_size]
					with tf.GradientTape() as tape:
						logits = self.model.call(current_inputs)
						loss = self.model.loss(logits, current_labels)
					gradients = tape.gradient(loss, self.model.trainable_variables)
					self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		print("Final testing...")
		test_inputs = []
		test_labels = []
		for i in range(0, len(self.testing_data)):
			test_inputs.append(self.testing_data[i].nodes)
			test_labels.append(self.testing_data[i].label)

		test_inputs = tf.convert_to_tensor(test_inputs, dtype='float32')
		test_labels = tf.convert_to_tensor(test_labels, dtype='int32')
		predictions = self.model.call(test_inputs)
		accuracy = self.model.accuracy(predictions, test_labels)
		print("Accuracy: ")
		print(accuracy)

	def save(self):
		print("Saving...")

	def test(self):
		print("Testing...")
		test_inputs = []
		test_labels = []
		for i in range(0, len(self.testing_data)):
			test_inputs.append(self.testing_data[i].nodes)
			test_labels.append(self.testing_data[i].label)

		test_inputs = tf.convert_to_tensor(test_inputs, dtype='float32')
		test_labels = tf.convert_to_tensor(test_labels, dtype='float32')
		predictions = self.model.call(test_inputs)
		accuracy = self.model.accuracy(predictions, test_labels)
		print("Accuracy: ")
		print(accuracy)

	def predict(self):
		pass


class DenseModel(tf.keras.Model):
	def __init__(self):

		super(DenseModel, self).__init__()

		self.batch_size = 16
		self.num_classes = 2
		self.number_of_epochs = 50
		self.number_of_examples = 458
		self.learning_rate = 0.0001
		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
		self.flatten = tf.keras.layers.Flatten()
		self.dense_one = tf.keras.layers.Dense(units=300, activation = 'relu', dtype='float32')
		self.dense_two = tf.keras.layers.Dense(units=150, activation = 'relu')
		self.dense_three = tf.keras.layers.Dense(units=1)

	def call(self, inputs):

		output_one = self.flatten(inputs)
		output_two = self.dense_one(output_one)
		output_three = self.dense_two(output_two)
		output_four = self.dense_three(output_three)

		logits = output_four

		return logits

	def loss(self, logits, labels):

		labels = tf.reshape(labels, [self.batch_size, -1])
		loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
		return loss

	def accuracy(self, logits, labels):

		correct_sum = 0.0
		probs = tf.math.sigmoid(logits)
		for i, prob in enumerate(probs):
			if prob < 0.5:
				if labels[i] == 0:
					correct_sum += 1
			else:
				if labels[i] == 1:
					correct_sum += 1
		return correct_sum / len(logits)
