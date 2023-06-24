import os
from lxml import etree
import json
import pandas as pd

import nltk
nltk.download('punkt')  # Download the necessary resources for tokenization
from nltk.tokenize import sent_tokenize


def text_preprocessing(text):
    text = text.replace(' ', ' ')
    text = text.replace('œ', 'oe')
    return text.replace('\n', ' ').strip()


def run_perdido(text, geoparser):
    try :
        return geoparser(text_preprocessing(text))
    except etree.XMLSyntaxError as e:
        return None
    

def get_term_occurrences_from_ene(filename):
    words = []
    if os.path.exists(filename):
        try:
            tree = etree.parse(filename)
            for term in tree.xpath('.//rs[@type="place" and @subtype="ene"]/term[@type="place"]'):
                phrase = ''
                tokens = term.xpath('.//w[@pos="N" or @pos="PREPDET" or @pos="PREP" or @pos="DET"]')
                for i, w in enumerate(tokens):
                    if len(w.text) > 1:
                        if ('DET' not in w.get('pos') and 'PREP' not in w.get('pos')):
                            phrase += w.text.lower() + ' '
                        if ('DET' in w.get('pos') or 'PREP' in w.get('pos')) and (i > 0 and i < len(tokens)-1):
                            phrase += w.text.lower() + ' '
                words.append(phrase.strip())
        except:
            pass
    return words


def get_ngrams_wt_term_outside_ene(filename, frequency_dict_geo, ngram_id, position=4, ngram_size=7):
    json_content = []
    print_content = ''
    if os.path.exists(filename):
        try:
            tree = etree.parse(filename)
            tokens = tree.xpath('.//w')
            for i, token in enumerate(tokens):
                print_content = ''
                if token.text in frequency_dict_geo:
                    line = {'num':ngram_id, 'class':'1', 'id_phrase':'0','pivot':token.text,'occurrence': '0', 'url':filename}
                    phrase = []
                    for j in range(position-1, 0, -1):
                        try:
                            words = {'word':tokens[i-j].text, 'POS':tokens[i-j].get('pos'), 'lemma':tokens[i-j].get('lemma')}
                            print_content += tokens[i-j].text + ' '
                        except IndexError:
                            words = {'word':'_', 'POS':'_', 'lemma':'_'}
                            print_content += '_ '
                        phrase.append(words)
                        
                    phrase.append({'word':token.text, 'POS':token.get('pos') + '+LS', 'lemma':token.get('lemma')})
                    print_content += '[ ' + token.text + ' ] '
                    for j in range(1, ngram_size+1-position):
                        try:
                            words = {'word':tokens[i+j].text, 'POS':tokens[i-j].get('pos'), 'lemma':tokens[i-j].get('lemma')}
                            print_content += tokens[i+j].text + ' '
                        except IndexError:
                            words = {'word':'_', 'POS':'_', 'lemma':'_'}
                            print_content += '_ '
                        phrase.append(words)
                        
                    print(print_content)
                    line['phrase'] = phrase
                    
                    ngram_id += 1
                    json_content.append(line)
        except:
            print("pass")
            pass
    
    return json_content


def load_lexicon(lexicon_filename):
    with open(lexicon_filename) as fp:
        return json.load(fp)


def load_edda_dataframe(edda_dataset_path, domain=None):
    data = pd.read_csv(edda_dataset_path, sep='\t')
    data.rename(columns={'edda-superdomainPred1':'domain'}, inplace=True)
    if domain:
        data = data[data['domain'] == domain]
    return data[['volume', 'numero', 'head', 'author','content', 'domain']]


def segment_sentences(text):
    sentences = sent_tokenize(text_preprocessing(text))
    return sentences