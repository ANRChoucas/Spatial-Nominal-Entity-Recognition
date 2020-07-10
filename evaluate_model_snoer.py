import argparse
import random
import pandas as pd
import numpy as np
import treetaggerwrapper
from keras.models import load_model
from gensim.models import fasttext
from joblib import load
from sklearn.decomposition import PCA


def sentences_to_ngrams(sentences, ngram_size, fr_nouns_file):

	ngrams = []
	context_size = int(ngram_size / 2)
	tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGINENC='utf-8', TAGOUTENC='utf-8')

	with open(fr_nouns_file, "r") as file:
		fr_nouns = file.readlines()

	for s in sentences:
		s = s.replace(';', '')
		s = s.replace("'", chr(39))
		s = s.replace('\'', chr(39))
		s = s.replace("d\'", " deeee ")
		s = s.replace("l\'", " leeee ")

		sentence_tagged = treetaggerwrapper.make_tags(tagger.tag_text(s))
		sentence = list(np.array(sentence_tagged)[:, 0])  # getting only the token (not lemmas and POS)

		for i, token in enumerate(sentence):
			if token == "leeee":
				sentence[i] = "l\'"
			if token == 'deeee':
				sentence[i] = "d\'"

		index_left = sentence.index('[')
		index_right = sentence.index(']')

		phrase_ngram = []

		# add left context
		for i in range(context_size):
			try:
				phrase_ngram.append(sentence[index_left - context_size + i])
			except IndexError:
				# when there is not enough words (ex: pivot word starting the sentence)
				phrase_ngram.append(random.choice(fr_nouns).rstrip())

		# add pivot token(s) (can contain several tokens)
		phrase_ngram.append(' '.join(sentence[index_left + 1:index_right]))

		# add right context
		for i in range(context_size):
			try:
				phrase_ngram.append(sentence[index_right + 1 + i])
			except IndexError:
				# when there is not enough words (ex: pivot word starting the sentence)
				phrase_ngram.append(random.choice(fr_nouns).rstrip())

		ngrams.append(phrase_ngram)

	return ngrams


def vectorization(ngram_size, input_data, we_vector_size, fasttext_wv):

	data_vec = np.array([])

	for phrase in input_data:
		phrase_vec = np.array([])

		for word in phrase:
			word = word.replace("’", "\'")
			vec = fasttext_wv[word]
			phrase_vec = np.append(phrase_vec, vec)

		data_vec = np.append(data_vec, phrase_vec)

	data_vec = np.reshape(data_vec, (len(input_data), ngram_size, we_vector_size))

	return data_vec


if __name__ == "__main__":

	# python3 ./evaluate_model_snoer.py -i "./data/corpus_validation.csv" -ti "./data/corpus_train.csv" -ft "./data/cc.fr.300.bin" -fr_nouns "./data/French_nouns.txt" -alg "GRU" -m "./models/GRU_5grams.h5" -n 5
	# python3 ./evaluate_model_snoer.py -i "./data/corpus_validation.csv" -ti "./data/corpus_train.csv" -ft "./data/cc.fr.300.bin" -fr_nouns "./data/French_nouns.txt" -alg "MLP_PCA" -m "./models/MLP_PCA_5grams.h5" -n 5
	# python3 ./evaluate_model_snoer.py -i "./data/corpus_validation.csv" -ti "./data/corpus_train.csv" -ft "./data/cc.fr.300.bin" -fr_nouns "./data/French_nouns.txt" -alg "MLP_AE" -m "./models/MLP_AE_5grams.h5" -n 5
	# python3 ./evaluate_model_snoer.py -i "./data/corpus_validation.csv" -ti "./data/corpus_train.csv" -ft "./data/cc.fr.300.bin" -fr_nouns "./data/French_nouns.txt" -alg "RF" -m "./models/RF_5grams.joblib" -n 5
	# python3 ./evaluate_model_snoer.py -i "./data/corpus_validation.csv" -ti "./data/corpus_train.csv" -ft "./data/cc.fr.300.bin" -fr_nouns "./data/French_nouns.txt" -alg "SVM" -m "./models/SVM_5grams.joblib" -n 5

	parser = argparse.ArgumentParser(description='Load ML model for spatial nominal entity recognition')
	parser.add_argument('-i', dest='input_data', required=True, help='input data')
	parser.add_argument('-n', dest='ngram_size', type=int, default=5, help='ngram size')
	parser.add_argument('-s', dest='we_vector_size', type=int, default=300, help='WE vector size')
	parser.add_argument('-m', '--model', dest='model_path', required=True, help='model filepath')
	parser.add_argument('-ti', '--train_input', dest='train_corpus_filepath', required=True, help='train input filepath')
	parser.add_argument('-ft', '--fasttext', dest='model_fasttext', required=True, help='fasttext model path')
	parser.add_argument('-fr_nouns', dest='fr_nouns_file', required=True, help='french nouns list filepath')
	parser.add_argument('-alg', dest='algorithm', required=True, help='name of the architecture: GRU, SVM, MLP_PCA, RF, MLP_AE')
	parser.add_argument('-v', '--verbose', dest='verbose', type=int, default=0, help='verbose mode')

	args = parser.parse_args()

	print('\n ** Load input data... \n')
	df = pd.read_csv(args.input_data, delimiter=';', names=['idf', 'labels', 'sentences', 'pivot_words', 'src', 'alea'])

	print(df.head(5))

	y_test = df['labels']

	print('\n ** Transform sentences to ' + str(args.ngram_size) + ' ngrams... \n')
	ngrams_list = sentences_to_ngrams(df['sentences'], args.ngram_size, args.fr_nouns_file)
	print(ngrams_list)

	print("\n ** Loading fastText model...\n")
	fasttext_model = fasttext.load_facebook_vectors(args.model_fasttext)

	print('\n ** Vectorisation of inputs... \n')
	x_test = vectorization(args.ngram_size, ngrams_list, args.we_vector_size, fasttext_model)

	np.random.seed(1)

	print('\n ** Loading model ' + args.model_path + ' \n')
	keras_models = ['GRU', 'MLP_PCA', 'MLP_AE']

	if args.algorithm in keras_models:
		clf = load_model(args.model_path)
	else:
		clf = load(args.model_path)

	print('\n ** Predicting... \n')

	if args.algorithm == 'RF' or args.algorithm == 'SVM' or args.algorithm == 'MLP_AE' or args.algorithm == 'MLP_PCA':
		x_test = np.reshape(x_test, (len(x_test), args.ngram_size * args.we_vector_size))

	if args.algorithm == 'MLP_PCA':
		df_train = pd.read_csv(args.train_corpus_filepath, delimiter=';', names=['idf', 'labels', 'sentences', 'pivot_words', 'alea'])

		ngrams_list_train = sentences_to_ngrams(df_train['sentences'], args.ngram_size, args.fr_nouns_file)
		x_train = vectorization(args.ngram_size, ngrams_list_train, args.we_vector_size, fasttext_model)
		x_train = np.reshape(x_train, (len(x_train), args.ngram_size * args.we_vector_size))

		# pca = PCA(0.99)
		if args.ngram_size == 1:
			pca = PCA(n_components=87, random_state=1)
		if args.ngram_size == 5:
			pca = PCA(n_components=295, random_state=1)
		if args.ngram_size == 7:
			pca = PCA(n_components=369, random_state=1)

		pca.fit(x_train)

		x_test = pca.transform(x_test)

	if args.algorithm in keras_models:
		# y_pred = clf.predict_classes(x_test)
		score = clf.evaluate(x_test, y_test)
		accuracy = score[1]

	if args.algorithm == 'RF' or args.algorithm == 'SVM':
		# y_pred = clf.predict(x_test)
		accuracy = clf.score(x_test, y_test)

	print('Test accuracy:', accuracy)
