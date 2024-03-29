import argparse
import os

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

if args.model == 'dense':
	import models.dense as dense
else:
	import models.graph as graph

## --------------------------------------------------------------------------------------

def prediction_mode(classifier):
	word_list = ['annoyed', 'relaxed', 'enthusiastic', 'calm', 'disappointed', 'aroused', 'neutral', 'sluggish', 'peppy', 'quiet', 'still', 'surprised', 'sleepy', 'nervous', 'afraid', 'satisfied', 'disgusted', 'angry', 'happy', 'sad']
	while True:
		subject = {}
		for word in word_list:
			x = input(word + ' x pos: ')
			y = input(word + ' y pos: ')
			subject[word_list.index(word)] = (x, y)
		print(classifier.predict(subject))
	pass

def main():
	# Initialize model
	if not (args.condition == 'depression' or args.condition == 'alex'):
		raise ValueError('Incorrect condition parameter')

	if not (args.mode == 'train' or args.mode == 'test' or args.mode == 'predict'):
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
	elif args.mode == 'predict':
		prediction_mode(classifier)


if __name__ == '__main__':
   main()
