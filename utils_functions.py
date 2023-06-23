import os
from lxml import etree


def text_preprocessing(text):
    text = text.replace(' ', ' ')
    text = text.replace('œ', 'oe')
    return text.replace('\n', ' ').strip()


def run_perdido(text, output_dir, filename, geoparser, ):
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


def get_ngrams_wt_term_outside_ene(filename, frequency_dict_geo, ngram_id):
    json_content = []
    if os.path.exists(filename):
        try:
            
            tree = etree.parse(filename)
            tokens = tree.xpath('.//w')
            for i, token in enumerate(tokens):
                if token.text in frequency_dict_geo:
                    line = {'num':ngram_id, 'class':'1', 'id_phrase':'0','pivot':token.text,'occurrence': '0', 'url':filename}
                    phrase = []
                    for j in range(3,0,-1):
                        try:
                            words = {'word':tokens[i-j].text, 'POS':tokens[i-j].get('pos'), 'lemma':tokens[i-j].get('lemma')}
                        except IndexError:
                            words = {'word':'_', 'POS':'_', 'lemma':'_'}
                        phrase.append(words)
                    phrase.append({'word':token.text, 'POS':token.get('pos') + '+LS', 'lemma':token.get('lemma')})
                    for j in range(1,4):
                        try:
                            words = {'word':tokens[i+j].text, 'POS':tokens[i+j].get('pos'), 'lemma':tokens[i+j].get('lemma')}
                        except IndexError:
                            words = {'word':'_', 'POS':'_', 'lemma':'_'}
                        phrase.append(words)
                    line['phrase'] = phrase
                    try:
                        print(tokens[i-3].text, tokens[i-2].text , tokens[i-1].text , '[', token.text, ']', tokens[i+1].text, tokens[i+2].text, tokens[i+3].text)
                    except IndexError:
                        pass
                    ngram_id += 1
                    json_content.append(line)
        except :
            pass
    
    return json_content


