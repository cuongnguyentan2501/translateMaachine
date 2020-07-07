from numpy import array
from numpy import asarray
from numpy import zeros
def wordEmbedding(MAX_NUM_WORDS,EMBEDDING_SIZE,word2idx_inputs):

    embeddings_dictionary = dict()

    glove_file = open(r'glove.6B/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()

        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
    for word, index in word2idx_inputs.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return  embedding_matrix,embeddings_dictionary,num_words