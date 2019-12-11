import argparse
import os
import models.dense as dense
import models.graph as graph

parser = argparse.ArgumentParser(description='Emotion Classifier')

parser.add_argument('--mode', type=str, default=None,
					help='Can be "train" or "test"')

parser.add_argument('--condition', type=str, default=None,
					help='Can be "depression" or "alex"')

parser.add_argument('--model', type=str, default=None,
					help='Can be "graph" or "dense"')

parser.add_argument('--name', type=str, default=None,
					help='Name of saved model')

args = parser.parse_args()

## --------------------------------------------------------------------------------------

def main():
	# Initialize model
	if not (args.condition == 'depression' or args.condition == 'alex'):
		raise ValueError('Incorrect condition parameter')

	if not (args.mode == 'train' or args.mode == 'test'):
		raise ValueError('Incorrect mode parameter')

	if not (args.model == 'graph' or args.model =='dense'):
		raise ValueError('Incorrect model parameter')

	classifier = None
	if args.model == 'graph':
		if args.condition == 'depression':
			classifier = graph.GraphDepressionClassifier(mode=args.mode, name=args.name)
		else:
			classifier = graph.GraphAlexClassifier(mode=args.mode, name=args.name)
	else:
		if args.condition == 'depression':
			classifier = dense.DenseDepressionClassifier(mode=args.mode, name=args.name)
		else:
			classifier = dense.DenseDepressionClassifier(mode=args.mode, name=args.name)

	if args.mode == 'train':
		classifier.train()
		classifier.save()
	elif args.mode == 'test':
		classifier.test()


if __name__ == '__main__':
   main()
