from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
import numpy as np
def sentence(senc,input_sentences,max_input_len,
             encoder_inputs_placeholder,encoder_states,decoder_embedding,decoder_lstm,decoder_dense,word2idx_outputs,
             MAX_NUM_WORDS,LSTM_NODES,max_out_len):
    senc=[senc]
    # token sequence to BagofWords
    tokenizer = ''
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(input_sentences)
    input_integer_seq = tokenizer.texts_to_sequences(senc)

    # ==========
    word2idx_inputs = tokenizer.word_index
    #print('Total unique words in the input: %s' % len(word2idx_inputs))
    num_words_output = len(word2idx_inputs) + 1
    #print("Length of longest sentence in input: %g" % max_input_len)
    # Padding sequence
    encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
    #print('encoder_input_sequences'+str(len(encoder_input_sequences)))
    encoder_model = Model(encoder_inputs_placeholder, encoder_states)
    decoder_state_input_h = Input(shape=(LSTM_NODES,))
    decoder_state_input_c = Input(shape=(LSTM_NODES,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs_single = Input(shape=(1,))
    decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
    decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs_single] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    idx2word_input = {v: k for k, v in word2idx_inputs.items()}
    idx2word_target = {v: k for k, v in word2idx_outputs.items()}

    def translate_sentence(input_seq):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))

        target_seq[0, 0] = word2idx_outputs['<sos>']
        eos = word2idx_outputs['<eos>']
        output_sentence = []

        for _ in range(max_out_len):
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            idx = np.argmax(output_tokens[0, 0, :])

            if eos == idx:
                break

            word = ''

            if idx > 0:
                word = idx2word_target[idx]
                output_sentence.append(word)

            target_seq[0, 0] = idx
            states_value = [h, c]

        return ' '.join(output_sentence)


    output=translate_sentence(encoder_input_sequences)



    return output
