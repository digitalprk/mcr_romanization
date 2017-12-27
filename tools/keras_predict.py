from keras.models import Model, load_model
from keras.layers import Input
from hanja import hangul
import pickle
import numpy as np
try:
    from jamo import decompose_character
except:
    from tools.jamo import decompose_character
    
latent_dim = 256
num_encoder_tokens = 174
num_decoder_tokens = 72
max_encoder_seq_length = 27
max_decoder_seq_length = 30


class Translit():
    
    def __init__(self, romanizer_model, input_token_index, target_token_index):
        self.model = romanizer_model
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index

        self.encoder = self.model.layers[2]
        self.encoder_inputs = self.model.layers[0].input
        self.encoder_outputs, state_h, state_c = self.model.layers[2].output
        self.encoder_states = [state_h, state_c]
        
        self.decoder_inputs = self.model.layers[1].input
        self.decoder_lstm = self.model.layers[3]
        self.decoder_dense = self.model.layers[-1]
        
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        
        self.decoder_state_input_h = Input(shape=(latent_dim,))
        self.decoder_state_input_c = Input(shape=(latent_dim,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=self.decoder_states_inputs)
        self.decoder_states = [state_h, state_c]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states)

        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())
        
    def __encode_input(self, name):
        test_input = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        index = 0
        for char in name:
            jamos = decompose_character(char)
            for jamo in jamos:
                if jamo not in self.input_token_index:
                    jamo = '-'
                test_input[0, index, self.input_token_index[jamo]] = 1.
                index += 1
        return test_input
    
    def __decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, self.target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
    
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if (sampled_char == '\n' or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]
    
        return decoded_sentence
    
    def Romanize(self, word):
        if len(word) > max_encoder_seq_length:
            return word
        if not hangul.contains_hangul(word):
            return word
        return self.__decode_sequence(self.__encode_input(word)).strip()