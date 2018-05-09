'''Sequence to sequence example in Keras (character-level).'''
from __future__ import print_function
import json
import math
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, Callback
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Average


def load_vocabularies(mode):

    input_vocabulary = {}
    output_vocabulary = {}

    if mode == "token":
        input_vocab_path = "data/processed/input_vocab_token_10000.json"
        output_vocab_path = "data/processed/output_vocab_token_10000.json"
    elif mode == "subtoken":
        input_vocab_path = "data/processed/input_vocab_subtoken_10000.json"
        output_vocab_path = "data/processed/output_vocab_subtoken_10000.json"

    with open(input_vocab_path, encoding='utf-8') as input_vocab_file:
        input_vocabulary = json.load(input_vocab_file)
    with open(output_vocab_path, encoding='utf-8') as output_vocab_file:
        output_vocabulary = json.load(output_vocab_file)

    return input_vocabulary, output_vocabulary


# Other valid values for mode will be "subtoken" and "character"
def data_generator(config, type_):
    
    filename = config[type_ + "_path"]
    input_vocabulary = config['input_vocabulary']
    output_vocabulary = config['output_vocabulary']
    max_encoder_seq_length = config['max_encoder_seq_length']
    max_decoder_seq_length = config['max_decoder_seq_length']
    num_encoder_tokens = config['num_encoder_tokens']
    num_decoder_tokens = config['num_decoder_tokens']
    mode = config['mode']
    batch_size = config['batch_size']
    num_contexts = config['num_contexts']
    context_size = config['context_size']

    lines_read = 0

    while True:

        # Reopen the file and re-read it in a loop.
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
                    usage = data['usage']
                    for ci, context in enumerate(usage):
                        if ci >= num_contexts:
                            break
                        encoder_inputs[ci][index_in_batch][0][input_vocabulary["<<START>>"]] = 1.
                        word_middle = math.ceil(len(context) / 2)
                        for wi, word in enumerate(
                            context[word_middle - context_size:word_middle + context_size],
                            start=1):
                            word_index = input_vocabulary.get(word, input_vocabulary["<<UNK>>"])
                            encoder_inputs[ci][index_in_batch][wi][word_index] = 1.
                        encoder_inputs[ci][index_in_batch][wi + 1][input_vocabulary["<<END>>"]] = 1.

                    decoder_input[index_in_batch][0][output_vocabulary["<<START>>"]] = 1.
                    for i, word in enumerate([data['variableName']], start=1):
                        word_index = output_vocabulary.get(word, output_vocabulary["<<UNK>>"])
                        decoder_input[index_in_batch][i][word_index] = 1.
                        if i > 0:
                            decoder_target[index_in_batch][i - 1][word_index] = 1.
                    decoder_target[index_in_batch][i][output_vocabulary["<<END>>"]] = 1.

                elif mode == "subtoken":

                    usage = data['usage']
                    for ci, context in enumerate(usage):
                        if ci >= num_contexts:
                            break
                        encoder_inputs[ci][index_in_batch][0][input_vocabulary["<<START>>"]] = 1.
                        before = context['before'][-context_size:]
                        after = context['after'][:context_size]
                        for wi, word in enumerate(before, start=1):
                            word_index = input_vocabulary.get(word, input_vocabulary["<<UNK>>"])
                            encoder_inputs[ci][index_in_batch][wi][word_index] = 1.
                        encoder_inputs[ci][index_in_batch][wi + 1][input_vocabulary["<<REF>>"]] = 1.
                        for wi, word in enumerate(after, start=(wi + 2)):
                            word_index = input_vocabulary.get(word, input_vocabulary["<<UNK>>"])
                            encoder_inputs[ci][index_in_batch][wi][word_index] = 1.
                        encoder_inputs[ci][index_in_batch][wi + 1][input_vocabulary["<<END>>"]] = 1.

                    decoder_input[index_in_batch][0][output_vocabulary["<<START>>"]] = 1.
                    for i, word in enumerate(data['variableName'], start=1):
                        if i >= max_decoder_seq_length - 1:
                            break
                        word_index = output_vocabulary.get(word, output_vocabulary["<<UNK>>"])
                        decoder_input[index_in_batch][i][word_index] = 1.
                        if i > 0:
                            decoder_target[index_in_batch][i - 1][word_index] = 1.
                    decoder_target[index_in_batch][i][output_vocabulary["<<END>>"]] = 1.

                if index_in_batch == batch_size - 1:
                    yield (all_inputs, decoder_target)

                lines_read += 1
        
            # This will flush the last partial batch
            yield (all_inputs, decoder_target)


def get_config(mode):

    config = { 'mode': mode }
    config['num_contexts'] = 5
    config['batch_size'] = 32
    config['epochs'] = 30
    config['latent_dim'] = 256
    config['train_size'] = 17497673
    config['validate_size'] = 200000  # 2495770
    config['test_size'] = 2164443
    config['train_batches'] = math.ceil(config['train_size'] / config['batch_size'])
    config['validate_batches'] = math.ceil(config['validate_size'] / config['batch_size'])
    config['test_batches'] = math.ceil(config['test_size'] / config['batch_size'])
    config['input_vocabulary'], config['output_vocabulary'] = load_vocabularies(mode)
    config['input_id_2_word'] = { v: k for k, v in config['input_vocabulary'].items() } 
    config['output_id_2_word'] = { v: k for k, v in config['output_vocabulary'].items() } 
    config['num_encoder_tokens'] = len(config['input_vocabulary'])
    config['num_decoder_tokens'] = len(config['output_vocabulary'])

    if mode == "token":
        config['context_size'] = 5
        config['train_path'] = "data/processed/train_shuffled.json"
        config['validate_path'] = "data/processed/validate_shuffled.json"
        config['test_path'] = "data/processed/test_shuffled.json"
        config['max_encoder_seq_length'] = config['context_size'] * 2 + 1 + 2
        config['max_decoder_seq_length'] = 1 + 2  # one character + start + stop

    elif mode == "subtoken":
        config['context_size'] = 8
        config['train_path'] = "data/processed/train_subtokens_shuffled.json"
        config['validate_path'] = "data/processed/validate_subtokens_shuffled.json"
        config['test_path'] = "data/processed/test_subtokens_shuffled.json"
        config['max_encoder_seq_length'] = config['context_size'] * 2 + 1 + 2
        config['max_decoder_seq_length'] = 4 + 2  # four subtokens + start + stop

    return config


def make_model(config):

    mode = config['mode']
    input_vocabulary = config['input_vocabulary']
    output_vocabulary = config['output_vocabulary']
    num_encoder_tokens = config['num_encoder_tokens']
    num_decoder_tokens = config['num_decoder_tokens']
    max_encoder_seq_length = config['max_encoder_seq_length']
    max_decoder_seq_length = config['max_decoder_seq_length']
    num_contexts = config['num_contexts']
    context_size = config['context_size']
    latent_dim = config['latent_dim']

    encoder_inputs_all = []

    # print('Number of unique input tokens:', num_encoder_tokens)
    # print('Number of unique output tokens:', num_decoder_tokens)
    # print('Max sequence length for inputs:', max_encoder_seq_length)
    # print('Max sequence length for outputs:', max_decoder_seq_length)

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
        encoder_inputs_all.append(encoder_inputs)
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
    return model, encoder_inputs_all, [h_state, c_state], decoder_lstm, decoder_inputs, decoder_dense


# Based on code example from https://stackoverflow.com/questions/43794995/python
class Checkpointer(Callback):

    def __init__(self, model, path, N):
        self.model = model
        self.N = N
        self.batch = 0
        self.path = path

    def on_batch_end(self, batch, logs={}):
        if self.batch > 0 and self.batch % self.N == 0:
            self.model.save_weights(self.path)
        self.batch += 1


if __name__ == "__main__":

    mode = sys.argv[1]
    print(mode)
    config = get_config(mode)
    model, _, _, _, _, _ = make_model(config)
    batch_checkpointer = Checkpointer(model, "models/" + mode + "_batch_weights.hdf5", 500)
    epoch_checkpointer = ModelCheckpoint(
        filepath="models/" + mode + "_weights.hdf5",
        verbose=1,
        save_best_only=True)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit_generator(
        generator=data_generator(config, 'train'),
        steps_per_epoch=config['train_batches'],
        verbose=1,
        validation_data=data_generator(config, 'validate'),
        validation_steps=config['validate_batches'],
        callbacks=[batch_checkpointer, epoch_checkpointer],
        epochs=config['epochs'])

    # Save model
    model.save(mode + ".h5")
