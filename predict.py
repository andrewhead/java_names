'''Sequence to sequence example in Keras (character-level).'''
from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Average

from rnn import make_model, data_generator, get_config


mode = 'subtoken'
config = get_config(mode)
model, encoder_inputs, encoder_state, decoder_lstm, decoder_inputs, decoder_dense = make_model(config)
model.load_weights("models/" + mode + "_batch_weights.hdf5")


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_state)

decoder_state_input_h = Input(shape=(config['latent_dim'],))
decoder_state_input_c = Input(shape=(config['latent_dim'],))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
input_vocabulary = config['input_vocabulary']
output_vocabulary = config['output_vocabulary']
reverse_input_char_index = {v: k for k, v in input_vocabulary.items()}
reverse_target_char_index = {v: k for k, v in output_vocabulary.items()}

num_decoder_tokens = config['num_decoder_tokens']
max_decoder_seq_length = config['max_decoder_seq_length']


def decode_sequence(input_sequences):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_sequences)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, output_vocabulary['<<START>>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        score = output_tokens[0, -1, sampled_token_index]
        print(sampled_char, score)
        print(output_tokens[0, -1, config['output_vocabulary']['<<END>>']])

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<<END>>' or
            (len(decoded) >= max_decoder_seq_length - 2) or
            score < 0.2):
            stop_condition = True
        else:
            # No unknowns allowed---always predict something
            if sampled_char == "<<UNK>>":
                output_tokens[0, -1, sampled_token_index] = 0
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sampled_token_index]
                score = output_tokens[0, -1, sampled_token_index]
            if score < 0.2:
                stop_condition = True
            else:
                decoded += [sampled_char]

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded


def render_target(tokens):
    target = ""
    for token in tokens:
        if len(target) > 0:
            target += token[0].upper() + token[1:]
        else:
            target += token
    return target


test_data = data_generator(config, 'validate')

for bi, batch in enumerate(test_data):

    for i in range(config['batch_size']):
        predict_inputs = []
        strings = []
        for j in range(config['num_contexts']):
            shape = batch[0][j][i].shape
            predict_inputs.append(batch[0][j][i].reshape((1, shape[0], shape[1])))
            input_sequence = np.argmax(batch[0][j][i], axis=1)
            isnt_padding = np.sum(batch[0][j][i]) > 0
            if isnt_padding:
                tokens = []
                for ci, c in enumerate(input_sequence):
                    token = reverse_input_char_index.get(c)
                    if token in ["<<START>>", "<<END>>"]:
                        continue
                    tokens.append(token)
                string = ' '.join(tokens)
                if len(string) > 0 and not string.isspace():
                    strings.append(string)
        # input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded = decode_sequence(predict_inputs)
        decoded_word = render_target(decoded)

        expected = ""
        for k in range(min(4, len(batch[1][i]))):
            if np.sum(batch[1][i][k] > 0):
                substr = reverse_target_char_index[np.argmax(batch[1][i][k])]
                if k > 0:
                    substr = substr[0].upper() + substr[1:]
                expected += substr

        if (len(decoded_word) > 0):
            message = "Expected: " + expected + ", Predicted: " + decoded_word
        else:
            message = "Expected: " + expected + ", NO GUESS."
        print()
        print(message)
        print("Contexts:")
        for s in strings:
            print("* " + s)
