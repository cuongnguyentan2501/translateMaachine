from keras.models import Model
from keras.layers import Input
import numpy as np
def predict(input_sentences,encoder_input_sequences,encoder_inputs_placeholder,encoder_states,decoder_embedding,
            decoder_lstm,decoder_dense,word2idx_inputs,word2idx_outputs,LSTM_NODES,max_out_len):
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
    idx2word_input = {v:k for k, v in word2idx_inputs.items()}
    idx2word_target = {v:k for k, v in word2idx_outputs.items()}
    def translate_sentence(input_seq):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))

        target_seq[0, 0] = word2idx_outputs['sos']
        eos = word2idx_outputs['eos']
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
    i = np.random.choice(len(input_sentences))
    input_seq = encoder_input_sequences[i:i+1]
    translation = translate_sentence(input_seq)
    print('-')
    print('Input:', input_sentences[i])
    print('Response:', translation)