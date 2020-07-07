def loadRawData(NUM_SENTENCES):
    input_sentences=[]
    output_sentences=[]
    output_sentences_inputs=[]
    count = 0
    for line in open(r'train-en-vi/train.en.txt', encoding="utf-8"):
        count += 1

        if count > NUM_SENTENCES or line=='':
            break
        input_sentence = line.strip()
        input_sentences.append(input_sentence)
    count=0
    for line in open(r'train-en-vi/train.vi.txt', encoding="utf-8"):

        count += 1
        if count > NUM_SENTENCES or line=='':
            break
        output = line.strip()
        output_sentence = output + ' <eos>'
        output_sentence_input = '<sos> ' + output
        output_sentences.append(output_sentence)
        output_sentences_inputs.append(output_sentence_input)
    return input_sentences,output_sentences,output_sentences_inputs