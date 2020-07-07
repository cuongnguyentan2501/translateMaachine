from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def tokenPadding(arrSumSequence,arrSequence,MAX_NUM_WORDS,input):
    #token sequence to BagofWords
    tokenizer=''
    if(input==True):
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    else:
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='')
    tokenizer.fit_on_texts(arrSumSequence)
    input_integer_seq = tokenizer.texts_to_sequences(arrSequence)
    #==========
    word2idx_inputs = tokenizer.word_index

    #print('Total unique words in the input: %s' % len(word2idx_inputs))
    num_words_output = len(word2idx_inputs) + 1
    max_input_len = max(len(sen) for sen in input_integer_seq)
    #print("Length of longest sentence in input: %g" % max_input_len)
    #Padding sequence
    if(input==True):
        encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
    else:
        encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len,padding='post')
    #===============
    return max_input_len,num_words_output,word2idx_inputs,encoder_input_sequences
