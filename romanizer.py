import pycrfsuite
import re
import pickle
from tools.vectorizer_crf import create_sentence_features_crf
from tools.keras_predict import Translit
from keras.models import load_model

class Romanizer():
    
    def __init__(self):
        self.__segmenter = pycrfsuite.Tagger()
        self.__segmenter.open('models/MCR-segmentation')
        
        romanizer_model = load_model('models/s2s.h5')
        with open('models/input_token_index.dat', 'rb') as fp:
            input_token_index = pickle.load(fp)
        with open('models/target_token_index.dat', 'rb') as fp:
            target_token_index = pickle.load(fp)
        
        self.__romanizer = Translit(romanizer_model, input_token_index, target_token_index)

    def Segment(self, text, return_array = False):
        
        sentences = re.split(r'(?<=\.) ', text)
        no_space_sentences = []
        segmented_sentences = []
        delimiters = []
        
        for sentence in sentences:
            no_space_sentence = ([c for c in ''.join(sentence.split())])
            no_space_sentences.append(no_space_sentence)
            delimiters.append(self.__segmenter.tag(create_sentence_features_crf(no_space_sentence)))
        
        for i, delimiter in enumerate(delimiters):
            current_sentence = ''
            for j, label in enumerate(delimiter):
                current_sentence += no_space_sentences[i][j]
                if label == '1' and j != len(delimiter) - 1:
                    current_sentence += ' '
            segmented_sentences.append(current_sentence)
        
        if return_array:
            return segmented_sentences
        
        return '. '.join(segmented_sentences).strip()
            
    def Romanize(self, text, return_array = False):
        sentences = self.Segment(text, return_array = True)
        romanized_sentences = []
        for sentence in sentences:
            words = sentence.split()
            romanized_sentence = ''
            for word in words:
                romanized_sentence += self.__romanizer.Romanize(word) + ' '
            romanized_sentences.append(romanized_sentence.strip(' '))
        
        if return_array:
            return romanized_sentences
        
        return '. '.join(romanized_sentences).strip()
        
                
            
    
    