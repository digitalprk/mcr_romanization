import sys
sys.path.append('../tools')

import pycrfsuite
from jamo import decompose_character
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
from hanja import hangul
import string
import pandas as pd

def get_jamos(character):
    if hangul.is_hangul(character):
        character_jamos = decompose_character(character, final_char = True)
    elif character in string.punctuation:
        character_jamos = ['.', '.', '.']
    elif character.isdigit():
        character_jamos = ['0', '0', '0']
    elif character.isalpha():
        character_jamos = ['a', 'a', 'a']
    else:
        character_jamos = ['x', 'x', 'x']
    return character_jamos

def character_features(sentence, index):
    character = sentence[index]
    character_jamos = get_jamos(character)

    features = ['bias',
                'char=' + character,
                'jamo1=' + character_jamos[0],
                'jamo2=' + character_jamos[1],
                'jamo3=' + character_jamos[2]]
    
    if index > 0:
        character_before_jamos = get_jamos(sentence[index - 1])
        features.extend(['before-char=' + sentence[index - 1],
                         'before-bigram=' + sentence[index - 1] + character,
                        'before-jamo1=' + character_before_jamos[0],
                        'before-jamo2=' + character_before_jamos[1],
                        'before-jamo3=' + character_before_jamos[2]])
    else:
        features.append('BOS')
    
    if index < len(sentence) - 1:
        character_after_jamos = get_jamos(sentence[index + 1])
        features.extend(['after-char=' + sentence[index + 1],
                         'after-bigram=' + character + sentence[index + 1],
                        'after-jamo1=' + character_after_jamos[0],
                        'after-jamo2=' + character_after_jamos[1],
                        'after-jamo3=' + character_after_jamos[2]])
    else:
        features.append('EOS')
    
    if index > 1:
        character_before_jamos = get_jamos(sentence[index - 2])
        features.extend(['before2-char=' + sentence[index - 2],
                         'before2-bigram=' + sentence[index - 2] + sentence[index - 1],
                         'before2-trigram=' + sentence[index - 2] + sentence[index - 1] + character,
                        'before2-jamo1=' + character_before_jamos[0],
                        'before2-jamo2=' + character_before_jamos[1],
                        'before2-jamo3=' + character_before_jamos[2]])
    else:
        features.append('BOS')
    
    if index < len(sentence) - 2:
        character_after_jamos = get_jamos(sentence[index + 2])
        features.extend(['after2-char=' + sentence[index + 2],
                         'after2-bigram=' + sentence[index + 1] + sentence[index + 2],
                         'after2-trigram=' + character + sentence[index + 1] + sentence[index + 2],
                        'after2-jamo1=' + character_after_jamos[0],
                        'after2-jamo2=' + character_after_jamos[1],
                        'after2-jamo3=' + character_after_jamos[2]])
    else:
        features.append('EOS')

    return features

def create_sentence_features(sentence):
    return [character_features(sentence, i) for i in range(len(sentence))]

def bio_classification_report(y_true, y_pred):

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

sentences = pd.read_csv('data/bibliographical_data.csv', encoding = 'utf8', sep = '\t', header = None)[0]

no_space_sentences = []
character_labels = []
for sentence in sentences:
    words = sentence.split()
    no_space_sentences.append([c for c in ''.join(words)])
    character_labels.append([str(_) for word in words for _ in (([0] * (len(word) - 1)) + [1])])

print('Vectorizing...')
X = [create_sentence_features(sentence) for sentence in no_space_sentences]
X_train, X_test, y_train, y_test = train_test_split(X, character_labels, test_size=0.3, random_state=777) 

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({'c1': 1.0,
                    'c2': 1e-3,
                    'max_iterations': 50,
                    'feature.possible_transitions': True})

print('Training...')
trainer.train('MCR-segmentation')

tagger = pycrfsuite.Tagger()
tagger.open('MCR-segmentation')

print('Evaluating...')
y_pred = [tagger.tag(xseq) for xseq in X_test]
print(bio_classification_report(y_test, y_pred))