from .classifier import Classifier
from .preprocess import get_data

import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GraphDepressionClassifier(Classifier):

	def __init__(self, mode, name):

		self.model_path = './graph_models/'

		print('Fetching Data...')
		self.training_data, self.testing_data = get_data('depression')
		print('Done.')

		self.name = name

		if mode == 'train':
			self.model = DepressionModel()
		elif mode == 'test':
			assert(False)
		else:
			assert(False)


	def train(self):

		print('Training...')
		for i in range(self.model.epochs):
			print('Epoch : ' + str(i))
			loss = nn.CrossEntropyLoss()

			loss_sum = 0
			loss_count = 0

			for i in range(0, len(self.training_data), self.model.batch_size):
				#gen batch of data
				batch = self.training_data[i: min(len(self.training_data), i + self.model.batch_size)]

				#generate batched_graph and coressponding labels
				graph_list = []
				labels = []

				for i, subject in enumerate(batch):
					graph_list.append(self.model.build_graph(subject))
					labels.append(subject.label)
				#batch graph
				batched_graph = dgl.batch(graph_list)
				#model step
				logits = self.model.forward(batched_graph)
				l = loss(logits, torch.tensor(labels))
				loss_sum += l
				loss_count += 1
				self.model.optimizer.zero_grad()
				l.backward()
				self.model.optimizer.step()

			print("Average Loss : " + str(loss_sum / loss_count))

		pass

	def save(self):

		file_path = self.model_path + self.name + '.pt'

		open(file_path, 'w+').close()

		torch.save(self.model.state_dict(), file_path)

	def test(self):

		graph_list = []
		labels = []

		print('Testing...')
		for i, subject in enumerate(self.testing_data):
			graph = self.model.build_graph(subject)
			graph_list.append(graph)
			labels.append(subject.label)
		batched_graph = dgl.batch(graph_list)
		logits = self.model.forward(batched_graph)
		print('Accuracy : ' + str(self.model.accuracy_function(logits, labels)))


class GraphAlexClassifier(Classifier):

	def __init__(self, mode, name):

		self.model_path = './graph_models/'

		print('Fetching Data...')
		self.training_data, self.testing_data = get_data('alex')
		print('Done.')

		self.name = name

		if mode == 'train':
			self.model = AlexModel()
		elif mode == 'test':
			assert(False)
		else:
			assert(False)

		pass

	def train(self):

		print('Training...')
		for i in range(self.model.epochs):
			print('Epoch : ' + str(i))
			loss = nn.CrossEntropyLoss()

			loss_sum = 0
			loss_count = 0

			for i in range(0, len(self.training_data), self.model.batch_size):
				#gen batch of data
				batch = self.training_data[i: min(len(self.training_data), i + self.model.batch_size)]

				#generate batched_graph and coressponding labels
				graph_list = []
				labels = []

				for i, subject in enumerate(batch):
					graph_list.append(self.model.build_graph(subject))
					labels.append(subject.label)
				#batch graph
				batched_graph = dgl.batch(graph_list)
				#model step
				logits = self.model.forward(batched_graph)
				l = loss(logits, torch.tensor(labels))
				loss_sum += l
				loss_count += 1
				self.model.optimizer.zero_grad()
				l.backward()
				self.model.optimizer.step()

			print("Average Loss : " + str(loss_sum / loss_count))

		pass

	def save(self):

		file_path = self.model_path + self.name + '.pt'

		open(file_path, 'a').close()

		torch.save(self.model.state_dict(), file_path)

		pass

	def test(self):

		graph_list = []
		labels = []

		print('Testing...')
		for i, subject in enumerate(self.testing_data):
			graph = self.model.build_graph(subject)
			graph_list.append(graph)
			labels.append(subject.label)
		batched_graph = dgl.batch(graph_list)
		logits = self.model.forward(batched_graph)
		print('Accuracy : ' + str(self.model.accuracy_function(logits, labels)))


class DepressionModel(nn.Module):
	"""
	Model class representing your MPNN. This is (almost) exactly like your Model
	class from TF2, but it inherits from nn.Module instead of keras.Model.
	Treat it identically to how you treated your Model class from previous
	assignments.
	"""

	def __init__(self):
		"""
		Init method for Model class. Instantiate a lifting layer, an optimizer,
		some number of MPLayers (we recommend 3), and a readout layer going from
		the latent space of message passing to the number of classes.
		"""
		super(DepressionModel, self).__init__()

		# Initialize hyperparameters
		self.raw_node_features = 2
		self.num_classes = 2
		self.batch_size = 16
		self.epochs = 10

		self.max_edge_dist = 30

		# Initialize trainable parameters
		self.node_lifting_layer = nn.Linear(in_features=self.raw_node_features, out_features=5, bias=True)
		self.mp_layer_one = MPLayer(in_feats=100, out_feats=100)
		self.mp_layer_two = MPLayer(in_feats=100, out_feats=100)
		self.mp_layer_three = MPLayer(in_feats=100, out_feats=100)

		self.readout_layer = nn.Linear(in_features=100, out_features=self.num_classes, bias=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

	def forward(self, g):
		"""
		Forward pass.
		"""
		nodefeatures = g.ndata.pop('nodefeatures')
		liftednodes = self.node_lifting_layer(nodefeatures)
		#Reshaping here worries me slightly
		liftednodes = torch.reshape(liftednodes, [liftednodes.size(0), -1])
		out_one = torch.relu(self.mp_layer_one(g, liftednodes))
		out_two = torch.relu(self.mp_layer_two(g, out_one))
		out_three = torch.relu(self.mp_layer_three(g, out_two))

		return self.readout(g, out_three)

	def readout(self, g, node_feats):
		"""
		Readout layer.
		"""
		g.ndata['nodefeatures'] = self.readout_layer(node_feats)
		return dgl.sum_nodes(g, 'nodefeatures')

	def accuracy_function(self, logits, labels):
		"""
		Computes the accuracy across a batch of logits and labels.

		:param logits: a 2-D np array of size (batch_size, 2)
		:param labels: a 1-D np array of size (batch_size)
									(1 for if the subject is active against cancer, else 0).
		:return: mean accuracy over batch.
		"""
		correct_sum = 0.0
		for i, logit in enumerate(logits):
			if logit[0] <= logit[1]:
				if labels[i] == 1:
					correct_sum += 1
			else:
				if labels[i] == 0:
					correct_sum += 1
		return correct_sum / len(logits)


	def build_graph(self, subject):
		"""
		Constructs a DGL graph out of a subject from the train/test data.

		:param subject: a subject object (see subject.py for more info)
		:return: A DGL Graph with the same number of nodes as atoms in the subject, edges connecting them,
				 and node features applied.
		"""

		graph = dgl.DGLGraph()
		graph.add_nodes(len(subject.nodes))
		data = np.zeros([len(subject.nodes), len(subject.nodes), 2])
		for i in range(len(subject.nodes)):
			data[i][i] = subject.nodes[i]

		graph.ndata['nodefeatures'] = torch.tensor(data, dtype=torch.float32)

		src_nodes = []
		dst_nodes = []

		for i, edge in enumerate(subject.edges):
			dx, dy = edge[1]
			if dy < self.max_edge_dist and dy > -self.max_edge_dist  and dx < self.max_edge_dist and dx > -self.max_edge_dist:
				src_nodes.append(edge[0][0])
				dst_nodes.append(edge[0][1])

		graph.add_edges(tuple(src_nodes), tuple(dst_nodes))

		return graph


class AlexModel(nn.Module):
	"""
	Model class representing your MPNN. This is (almost) exactly like your Model
	class from TF2, but it inherits from nn.Module instead of keras.Model.
	Treat it identically to how you treated your Model class from previous
	assignments.
	"""

	def __init__(self):
		"""
		Init method for Model class. Instantiate a lifting layer, an optimizer,
		some number of MPLayers (we recommend 3), and a readout layer going from
		the latent space of message passing to the number of classes.
		"""
		super(AlexModel, self).__init__()

		# Initialize hyperparameters
		self.raw_node_features = 2
		self.num_classes = 2
		self.batch_size = 16
		self.epochs = 10

		self.max_edge_dist = 200

		# Initialize trainable parameters
		self.node_lifting_layer = nn.Linear(in_features=self.raw_node_features, out_features=5, bias=True)
		self.mp_layer_one = MPLayer(in_feats=100, out_feats=100)
		self.mp_layer_two = MPLayer(in_feats=100, out_feats=100)
		self.mp_layer_three = MPLayer(in_feats=100, out_feats=100)

		self.readout_layer = nn.Linear(in_features=100, out_features=self.num_classes, bias=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

	def forward(self, g):
		"""
		Forward pass.
		"""
		nodefeatures = g.ndata.pop('nodefeatures')
		liftednodes = self.node_lifting_layer(nodefeatures)
		#Reshaping here worries me slightly
		liftednodes = torch.reshape(liftednodes, [liftednodes.size(0), -1])
		out_one = torch.relu(self.mp_layer_one(g, liftednodes))
		out_two = torch.relu(self.mp_layer_two(g, out_one))
		out_three = torch.relu(self.mp_layer_three(g, out_two))

		return self.readout(g, out_three)

	def readout(self, g, node_feats):
		"""
		Readout layer.
		"""
		g.ndata['nodefeatures'] = self.readout_layer(node_feats)
		return dgl.sum_nodes(g, 'nodefeatures')

	def accuracy_function(self, logits, labels):
		"""
		Computes the accuracy across a batch of logits and labels.

		:param logits: a 2-D np array of size (batch_size, 2)
		:param labels: a 1-D np array of size (batch_size)
									(1 for if the subject is active against cancer, else 0).
		:return: mean accuracy over batch.
		"""
		correct_sum = 0.0
		for i, logit in enumerate(logits):
			if logit[0] <= logit[1]:
				if labels[i] == 1:
					correct_sum += 1
			else:
				if labels[i] == 0:
					correct_sum += 1
		return correct_sum / len(logits)


	def build_graph(self, subject):
		"""
		Constructs a DGL graph out of a subject from the train/test data.

		:param subject: a subject object (see subject.py for more info)
		:return: A DGL Graph with the same number of nodes as atoms in the subject, edges connecting them,
				 and node features applied.
		"""

		graph = dgl.DGLGraph()
		graph.add_nodes(len(subject.nodes))
		data = np.zeros([len(subject.nodes), len(subject.nodes), 2])
		for i in range(len(subject.nodes)):
			data[i][i] = subject.nodes[i]

		graph.ndata['nodefeatures'] = torch.tensor(data, dtype=torch.float32)

		src_nodes = []
		dst_nodes = []

		for i, edge in enumerate(subject.edges):
			dx, dy = edge[1]
			if dy < self.max_edge_dist and dy > -self.max_edge_dist  and dx < self.max_edge_dist and dx > -self.max_edge_dist:
				src_nodes.append(edge[0][0])
				dst_nodes.append(edge[0][1])

		graph.add_edges(tuple(src_nodes), tuple(dst_nodes))

		return graph



class MPLayer(nn.Module):

	def __init__(self, in_feats, out_feats):
		super(MPLayer, self).__init__()

		self.forward_layer = nn.Linear(in_features=in_feats, out_features=out_feats)
		self.message_layer = nn.Linear(in_features=in_feats, out_features=out_feats)

		pass

	def forward(self, g, node_feats):
		"""
		Forward pass
		"""
		g.ndata['nodefeatures'] = node_feats

		g.send(message_func=self.message)

		g.recv(reduce_func=self.reduce)

		features = g.ndata.pop('nodefeatures')

		out_1 = self.forward_layer(features)

		return out_1

	def message(self, edges):
		"""
		Message function.
		"""
		src_nodes = edges.src['nodefeatures']
		out = self.message_layer(src_nodes)
		map = {}
		map['you_have_mail'] = torch.relu(out)
		return map


	def reduce(self, nodes):
		"""
		Reduction function.
		"""
		messages = nodes.mailbox['you_have_mail']
		summed_messages = torch.zeros([messages.shape[0], messages.shape[2]], dtype=torch.float32)
		for i, message in enumerate(messages):
			summed_message = torch.sum(message, dim=0)
			summed_messages[i] = summed_message
		return {'nodefeatures': summed_messages}
