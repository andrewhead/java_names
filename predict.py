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


def score_sequence(sequence):
    """
    Compute a score for a predicted sequence.  Each element in the sequence
    is a tuple, including a token ID and a score.
    """
    # Sum up the log scores of each of the components
    score_log_sum = 0
    for (_, score) in sequence:
        # To avoid overflow, if the score is 0, set it to a very small value.
        if score == 0:
            score = 1e-10
        score_log_sum += np.log(score)

    # Use the heuristic normalization described in Neubig's 2017 tutorial.
    length = len(sequence)
    score_normalized = score_log_sum / length
    return score_normalized


def beam_search(config, decoder, start_state, beam_size):
    """
    Do a beam search for promising expansions of the variable name.
    Args:
    * config     : configuration settings
    * decoder    : RNN decoder
    * start_state: output of encoder on some contexts
    * beam_size  : the number of options to keep
    Returns: List of tuples.  In each tuple is a sequence of scored token
    and a score for the whole sequence.
    """
    output_vocabulary = config['output_vocabulary']
    max_prediction_tokens = config['max_prediction_tokens']

    # The first sequence is just the start token.
    best_sequences = [[(output_vocabulary["<<START>>"], 1)]]

    # Redo the search to a new level of depth with the best sequences
    # seen so far, expanding them.
    sequence_length = 1
    while sequence_length < max_prediction_tokens:

        # Add new candidate sequences by expanding the current sequences
        candidate_sequences = list(best_sequences)
        for sequence in best_sequences:

            # Expand sequences at the frontier of the search.
            if len(sequence) == sequence_length:

                # Predict the next most likely tokens for this sequence
                token_ids = [int(s[0]) for s in sequence]
                next_tokens = get_next(
                    config, decoder, start_state, token_ids, beam_size)

                # Create new sequences for the expected tokens.
                # Ignore all "END" tokens, because this will often duplicate
                # sequences (one version with END, one version without).
                for next_token in next_tokens:
                    next_token_id = next_token[0]
                    if next_token_id != output_vocabulary["<<END>>"]:
                        candidate_sequence = sequence + [next_token]
                        candidate_sequences.append(candidate_sequence)

            # Remove all length-1 sequences from the beam, as they will
            # only contain a start sequence.
            if len(sequence) == 1:
                del(candidate_sequences[0])

        # Prune back to the top N best sequences
        sorted_sequences = sorted(
            candidate_sequences,
            key=lambda s: score_sequence(s),
            reverse=True)
        best_sequences = sorted_sequences[:beam_size]

        sequence_length += 1

    result = [(seq, score_sequence(seq)) for seq in best_sequences]
    return result


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


def render_output(config, token_ids):

    output_id_2_word = config['output_id_2_word']

    target = ""
    for tok_id in token_ids:
        word = output_id_2_word[tok_id]
        if word in ["<<START>>", "<<END>>"]:
            continue
        if len(target) > 0:
            target += word[0].upper() + word[1:]
        else:
            target += word
    return target


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
    parser.add_argument(
        "-b",
        help="number of batches to compute results for",
        type=int,
        default=10)
    parser.add_argument(
        "--beam-size",
        help="size of beam when predicting results",
        type=int,
        default=10)
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="whether to print variable context with each prediction")
    parser.add_argument(
        "--output-file",
        type=str,
        help="path to a file to save results in TSV format")
    args = parser.parse_args()
    unit = args.u
    weights_type = args.c
    batches = args.b
    beam_size = args.beam_size
    print_context = args.print_context
    output_filename = args.output_file

    if output_filename:
        output_file = open(output_filename, "w", encoding="utf-8")

    # Prepare configurable settings
    config = get_config(unit)
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
    test_data = data_generator(config, 'test')

    # Counters for summary statistics of accuracy
    num_unk = 0
    total_num = 0
    num_correct = 0

    for bi, batch in enumerate(test_data):

        # Enable the following to just print out a few lines of data
        if bi >= batches:
            break

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

            # Predict the output sequence with beam search.
            state = encoder_model.predict(predict_inputs)
            candidates = beam_search(config, decoder_model, state, beam_size)
            decoded_candidates = []
            for rank, candidate in enumerate(candidates, start=1):
                decoded_word = render_output(config, [t[0] for t in candidate[0]])
                decoded_candidates.append(decoded_word)
                if output_filename:
                    output = [expected, decoded_word, rank, candidate[1]]
                    if print_context:
                        output.extend(strings)
                    output_file.write('\t'.join([str(_) for _ in output]) + "\n")

            # Update counts for summary statistics
            total_num += 1
            if len(decoded_word) > 0 and expected in decoded_candidates:
                num_correct += 1
            if "<<UNK>>" in expected:
                num_unk += 1

        print(
            "Correctly predicted", num_correct, "/", total_num - num_unk,
            "known tokens (", total_num, "total )")
