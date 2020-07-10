# Spatial Nominal Entity Recognition


## Overview

This repository contains the source code for evaluating ML models trained for Spatial Nominal Entity Recognition as proposed in 

> Amine Medad, Mauro Gaio, Ludovic Moncla, Sébastien Mustière, and Yannick Le Nir. Comparing supervised learning algorithms for Spatial Nominal Entity recognition. The 23rd AGILE International Conference on Geographic Information Science. 2020


Datasets are given in the `corpus` directory and models in the `models` directory. 


## Installation

Install the required python librairies:

	pip3 install -r requirements.txt

Then you need to download the binary file of the pretrained French [FastText model](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz) (4.2 Go) and add it to the `data` directory:

	wget -P data https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz
	gzip -d data/cc.fr.300.bin.gz


[TreeTagger](https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/) needs also to be installed with the French parameter file before running the script:
	
	mkdir TreeTagger
	cd TreeTagger
	
	wget https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.3.tar.gz
	tar -xzf tree-tagger-linux-3.2.3.tar.gz

	wget https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
	tar -xzf tagger-scripts.tar.gz

	wget https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh
	bash install-tagger.sh

	cd lib/
	wget https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french.par.gz
	gunzip french.par.gz



## Usage

	python3 evaluate_model_snoer.py -i <input_dataset> -n <ngram_size> -alg <algorithm_name> -m <model_filepath> -ft <fasttext_model> -fr_nouns <french_nouns_filepath> -s <we_size_vec> -ti <train_dataset>

 * `<input_dataset>`: filepath to the csv input data
 * `<train_dataset>`: filepath to the csv training data (use for PCA fitting for the model MLP+PCA only)
 * `<fasttext_model>`: filepath of the pretrained FastText binary model
 * `<french_nouns_filepath>`: filepath of the file containing French nouns (use for padding ngrams)
 * `<algorithm_name>`: name of the architecture used for training (GRU, MLP+AE, MLP+PCA, SVM, RF)
 * `<model_filepath>`: filepath of the model to evaluate
 * `<ngram_size>`: size of the ngram (1, 5 or 7)
 * `<we_size_vec>`: Word Embedding dimension (default: 300)

 You can also download and execute the jupyter notebook version.

## Example

Run the following command to evaluate the GRU model trained with 5 grams :

    python3 evaluate_model_snoer.py -i "./data/corpus_validation.csv" -n 5 -alg "GRU" -m "./models/GRU_5grams.h5" -ft "./data/cc.fr.300.bin" -fr_nouns "./data/French_nouns.txt" -ti "./data/corpus_train.csv" 
    

## Results


<table>
  <tr>
    <td>Model</td>
    <td colspan="3">GRU</td>
    <td colspan="3">RF</td>
    <td colspan="3">SVM</td>
  </tr>
  <tr>
    <td>ngram_size</td>
    <td>1 g</td>
    <td>5 g</td>
    <td>7 g</td>
    <td>1 g</td>
    <td>5 g</td>
    <td>7 g</td>
    <td>1 g</td>
    <td>5 g</td>
    <td>7 g</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>0.67</td>
    <td>0.76</td>
    <td>0.79</td>
    <td>0.71</td>
    <td>0.73</td>
    <td>0.74</td>
    <td>0.69</td>
    <td>0.75</td>
    <td>0.72</td>
  </tr>
</table>

<table>
  <tr>
    <td>Model</td>
    <td colspan="3">MLP + AE</td>
    <td colspan="3">MLP + PCA</td>
  </tr>
  <tr>
    <td>ngram_size</td>
    <td>1 g</td>
    <td>5 g</td>
    <td>7 g</td>
    <td>1 g</td>
    <td>5 g</td>
    <td>7 g</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>0.68</td>
    <td>0.75</td>
    <td>0.78</td>
    <td>0.49</td>
    <td>0.64</td>
    <td>0.60</td>
  </tr>
</table>



## Acknowledgement

This work is supported and funded in part by French National Research Agency (ANR) under the CHOUCAS project (ANR-16-CE23-0018). 

The [CHOUCAS project](http://choucas.ign.fr) is a French interdisciplinary research project aiming to respond to a need expressed by the high mountain gendarmerie platoon to help localising victims in mountain area.
