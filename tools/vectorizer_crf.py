import string

from hanja import hangul
try:
    from jamo import decompose_character
except:
    from tools.jamo import decompose_character

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

def create_sentence_features_crf(sentence):
    return [character_features(sentence, i) for i in range(len(sentence))]