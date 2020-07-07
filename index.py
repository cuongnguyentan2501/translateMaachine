from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint
import numpy as np
import loadData
import tokenPadding
import wordEmbedding
import predict
import predictWithSentence
#==============================
BATCH_SIZE = 25
EPOCHS = 50
LSTM_NODES =256
NUM_SENTENCES = 5000
MAX_SENTENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100
checkpoint_path = "train-wight.ckpt"
input_sentences,output_sentences,output_sentences_inputs=loadData.loadRawData(NUM_SENTENCES)

#============================
#TokenAndPaddingWordSequences
max_input_len,_,word2idx_inputs,encoder_input_sequences=tokenPadding.tokenPadding(input_sentences,input_sentences,MAX_NUM_WORDS,True)
max_out_len,num_words_output,word2idx_outputs,decoder_output_sequences=tokenPadding.tokenPadding(output_sentences+output_sentences_inputs,output_sentences,MAX_NUM_WORDS,False)
_,_,_,decoder_input_sequences=tokenPadding.tokenPadding(output_sentences+output_sentences_inputs,output_sentences_inputs,MAX_NUM_WORDS,False)


#===========================
#EmbeddingSequenceUsingGlove
embedding_matrix,embeddings_dictionary,num_words=wordEmbedding.wordEmbedding(MAX_NUM_WORDS,EMBEDDING_SIZE,word2idx_inputs)

embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)
decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)

#=============================
#define model
#define encoder
for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1
encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]
#define decoder
decoder_inputs_placeholder = Input(shape=(max_out_len,))
decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

model.load_weights(checkpoint_path)

#===========================================
#Training
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,

                              verbose=1)
r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=[cp_callback])


predict.predict(input_sentences,encoder_input_sequences,encoder_inputs_placeholder,encoder_states,decoder_embedding,decoder_lstm,
                decoder_dense,word2idx_inputs,word2idx_outputs,LSTM_NODES,max_out_len)
predictWithSentence.sentence()

