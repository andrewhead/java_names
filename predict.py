'''Sequence to sequence example in Keras (character-level).'''
from __future__ import print_function
import sys
import argparse

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Average

from rnn import make_model, data_generator, get_config


def get_next(config, decoder, start_state, sequence, top):
    """
    Get the `top` number of next tokens in a sequence.
    Args:
    * config:      shared configuration object for training and prediction.
    * decoder:     a decoder
    * start_state: a vector of the starting state of the model
                   (should be the output of the encoder)
    * sequence:    sequence of token IDs in the sequence before
                   the one we're trying to predict next.  It's
                   expected that the first token in the sequence will always
                   be the start token.
    * output_vocabulary: 
                   map from token strings to numerical IDs
    * top:         how many options of next tokens to provide

    Returns: List of tuples, where each tuple is a token ID and a score
    (probability-ish) for that token.  Unknown is *never* included in the list.
    """
    output_vocabulary = config['output_vocabulary']
    num_decoder_tokens = config['num_decoder_tokens']
    UNK_ID = output_vocabulary["<<UNK>>"]
    states_value = start_state
    predictions = []

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    for tok_id in sequence:

        # Make a 1-length input token of the current expected token
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, tok_id] = 1.

        # Feed-forward through the decoder
        output_tokens, h, c = decoder.predict(
            [target_seq] + states_value)

        # Update states with the output of the decoder
        states_value = [h, c]

    # Force the probability of an unknown token to 0---we don't want to predict
    # the unknown, as it won't be useful to a user.
    output_tokens[0, -1, output_vocabulary["<<UNK>>"]] = 0.
  
    # Get the token IDs with the highest probabilities.
    # Store these IDs with their scores.
    top_tok_ids = output_tokens[0, -1, :].argsort()[-top:][::-1]
    for tok_id in top_tok_ids:
        predictions.append((tok_id, output_tokens[0, -1, tok_id]))

    return predictions


def get_state(encoder, input_sequences):
    """
    Get the start state for a decoder.
    Args
    * encoder        : an encoder
    * input_sequences: array of matrixes, each representing an input context.
    Returns: tuple of h and c state vectors
    """
    # Encode the input as state vectors.
    return encoder_model.predict(input_sequences)


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

        # Exit condition: either hit max length or find stop character.
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


def load_weights(model, unit, weights_type):
    if weights_type == "epoch":
        model.load_weights("models/" + unit + "_weights.hdf5")
    elif weights_type == "batch":
        model.load_weights("models/" + unit + "_batch_weights.hdf5")


def prepare_models(config, unit, weights_type):

    model, encoder_inputs, encoder_state, decoder_lstm, decoder_inputs, decoder_dense = make_model(config)
    load_weights(model, unit, weights_type)

    # Encoder is just the bottom of the model
    encoder = Model(encoder_inputs, encoder_state)

    # Define decoder model
    decoder_state_input_h = Input(shape=(config['latent_dim'],))
    decoder_state_input_c = Input(shape=(config['latent_dim'],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder, decoder


def context_to_string(config, context):
    """ Take 3D context and a string of approximately what it looked like. """

    input_vocabulary = config['input_id_2_word']
    sentence = ""

    # Get the token IDs from this context
    tok_ids = context[0, :, :].argmax(axis=1)

    # Add each word for each token to the sentence
    for tok_id in tok_ids:
        word = input_vocabulary[tok_id]
        sentence += " " + word

    return sentence


def get_contexts(config, batch):
    """
    Reshape a batch into a list of examples, each of which is comprised of
    multiple contexts.  This is a pre-processing step to make it easier to
    feed individual examples to the encoder.
    """
    contexts = []
    example_shape = batch[0][0][0].shape

    for i in range(config['batch_size']):

        example_contexts = []
        for j in range(config['num_contexts']):
            example_contexts.append(
                batch[0][j][i].reshape(
                    (1, example_shape[0], example_shape[1])))

        contexts.append(example_contexts)

    return contexts


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description="Use NN to predict variable names")
    parser.add_argument(
        "-u",
        help="unit",
        choices=["token", "subtoken"],
        required=True)
    parser.add_argument(
        "-c",
        help="type of checkpoint",
        choices=["epoch", "batch"],
        required=True)
    args = parser.parse_args()
    unit = args.u
    weights_type = args.c

    # Prepare configurable settings
    config = get_config(mode)
    input_vocabulary = config['input_vocabulary']
    output_vocabulary = config['output_vocabulary']
    num_decoder_tokens = config['num_decoder_tokens']
    max_decoder_seq_length = config['max_decoder_seq_length']

    # Reverse-lookup token index to decode sequences back to something readable
    reverse_input_char_index = config['input_id_2_word']
    reverse_target_char_index = config['output_id_2_word']

    # Load up the models
    model, encoder_model, decoder_model = prepare_models(config, unit, weights_type)

    # Set up the data generator
    test_data = data_generator(config, 'validate')

    # Counters for summary statistics of accuracy
    num_uncertain = 0
    total_num = 0
    num_correct = 0

    for bi, batch in enumerate(test_data):

        # Enable the following to just print out a few lines of data
        if bi > 10:
            break

        # Enable for computing accuracy
        # if bi >= 1563:
        #     break

        # Iterate over each test point separately
        for i in range(config['batch_size']):

            predict_inputs = []
            strings = []

            for j in range(config['num_contexts']):

                # Queue up another context as input data
                shape = batch[0][j][i].shape
                predict_inputs.append(
                    batch[0][j][i].reshape((1, shape[0], shape[1])))

                # Make a string for the context, for debug output.
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

            # Predict the output sequence.
            # This is the part where we should do a beam search.
            decoded = decode_sequence(predict_inputs)
            decoded_word = render_target(decoded)

            # Convert the expected string to a camel-case variable.
            # TODO(andrewhead): merge with `render_target`?
            expected = ""
            for k in range(len(batch[1][i]) - 1):
                if np.sum(batch[1][i][k] > 0):
                    substr = reverse_target_char_index[np.argmax(batch[1][i][k])]
                    if substr == "<<END>>":
                        break
                    if k > 0:
                        substr = substr[0].upper() + substr[1:]
                    expected += substr

            # If a non-unknown token is predicted, then print.
            if len(decoded_word) > 0:
                print("\t".join([
                    decoded_word,
                    expected,
                    "",
                    "",
                    "",
                    ] + strings
                ))

            # Update counts for summary statistics
            total_num += 1
            if len(decoded_word) == 0:
                num_uncertain += 1
            if len(decoded_word) > 0 and decoded_word == expected:
                num_correct += 1

    # print("Total:", total_num)
    # print("Predicted:", total_num - num_uncertain)
    # print("Correct of predicted:", num_correct)
