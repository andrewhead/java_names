'''Sequence to sequence example in Keras (character-level).

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and correspding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
'''
from __future__ import print_function
import json
import math

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Average

# train_size = 17497673
# validate_size = 2495770
test_size = 2164443
train_size = 100000
validate_size = 10000

num_contexts = 5  # Number of contexts to use for training
token_context_size = 5

batch_size = 32  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
train_batches = math.ceil(train_size / batch_size)
validate_batches = math.ceil(validate_size / batch_size)
test_batches = math.ceil(test_size / batch_size)


def load_vocabularies(mode="token"):

    input_vocabulary = {}
    output_vocabulary = {}

    if mode == "token":
        input_vocab_path = "data/processed/input_vocab_token.json"
        output_vocab_path = "data/processed/output_vocab_token.json"

    with open(input_vocab_path, encoding='utf-8') as input_vocab_file:
        input_vocabulary = json.load(input_vocab_file)
    with open(output_vocab_path, encoding='utf-8') as output_vocab_file:
        output_vocabulary = json.load(output_vocab_file)

    return input_vocabulary, output_vocabulary


# Other valid values for mode will be "subtoken" and "character"
def data_generator(filename, input_vocabulary, output_vocabulary, max_encoder_seq_length,
        max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, batch_size, mode="token"):

    lines_read = 0

    with open(filename, 'r', encoding='utf-8') as file_:

        for line in file_:

            index_in_batch = lines_read % batch_size

            # Set up new one-hot training set whenever we start a new batch
            # Do we need to add in the pad token?  Or is it just assumed that straight zeros
            # means that this is a padding token?
            if index_in_batch == 0:
                all_inputs = []
                encoder_inputs = []
                for _ in range(num_contexts):
                    encoder_input = np.zeros(
                        (batch_size, max_encoder_seq_length, num_encoder_tokens),
                        dtype='float32')
                    encoder_inputs.append(encoder_input)
                    all_inputs.append(encoder_input)
                decoder_input = np.zeros(
                    (batch_size, max_decoder_seq_length, num_decoder_tokens),
                    dtype='float32')
                all_inputs.append(decoder_input)
                decoder_target = np.zeros(
                    (batch_size, max_decoder_seq_length, num_decoder_tokens),
                    dtype='float32')

            # Load in the data for this line
            data = json.loads(line)

            if mode == "token":

                # Save one-hot values for encoder input
                # TODO: use multiple contexts.
                usage = data['usage']
                for ci, context in enumerate(usage):
                    if ci >= num_contexts:
                        break
                    encoder_inputs[ci][index_in_batch][0][input_vocabulary["<<START>>"]] = 1.
                    word_middle = math.ceil(len(context) / 2)
                    for wi, word in enumerate(
                        context[word_middle - token_context_size:word_middle + token_context_size],
                        start=1):
                        word_index = input_vocabulary.get(word, input_vocabulary["<<UNK>>"])
                        encoder_inputs[ci][index_in_batch][wi][word_index] = 1.
                    encoder_inputs[ci][index_in_batch][wi + 1][input_vocabulary["<<END>>"]] = 1.

                # TODO make this into sources instead of input
                decoder_input[index_in_batch][0][output_vocabulary["<<START>>"]] = 1.
                for i, word in enumerate([data['variableName']], start=1):
                    word_index = output_vocabulary.get(word, output_vocabulary["<<UNK>>"])
                    decoder_input[index_in_batch][i][word_index] = 1.
                    if i > 0:
                        decoder_target[index_in_batch][i - 1][word_index] = 1.

            if lines_read > 0 and index_in_batch == 0:
                yield (all_inputs, decoder_target)

            lines_read += 1
    
        # This will flush the last partial batch
        yield ([encoder_input, decoder_input], decoder_target)
        yield StopIteration


input_vocabulary, output_vocabulary = load_vocabularies(mode="token")

num_encoder_tokens = len(input_vocabulary)
num_decoder_tokens = len(output_vocabulary)

mode = "token"
if mode == "token":

    max_encoder_seq_length = token_context_size * 2 + 1 + 2
    max_decoder_seq_length = 1 + 2  # one character + start + stop

    train_path = "data/processed/train_shuffled.json"
    validate_path = "data/raw/validate_output.json"
    test_path = "data/raw/test_output.json"

print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


h_states = []
c_states = []
inputs = []


# Make one encoder for each context.
for _ in range(num_contexts):

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # Save the encoder inputs and outputs to a list
    inputs.append(encoder_inputs)
    h_states.append(state_h)
    c_states.append(state_c)


# Merge the output of each of the context LSTMs into one.
# TODO(andrew): Consider using another type of merge, like maximum.
h_state = Average()(h_states)
c_state = Average()(c_states)


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=[h_state, c_state])
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
inputs.append(decoder_inputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs, decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit_generator(
    generator=data_generator(train_path, input_vocabulary, output_vocabulary, max_encoder_seq_length,
        max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, batch_size, mode=mode),
    steps_per_epoch=train_batches,
    verbose=1,
    validation_data=data_generator(validate_path, input_vocabulary, output_vocabulary, max_encoder_seq_length,
        max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, batch_size, mode=mode),
    validation_steps=validate_batches,
    epochs=epochs)


"""
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
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
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
"""
