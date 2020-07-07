from tkinter import Tk, W, E,Text,Label,filedialog
from tkinter.ttk import Frame, Button, Style
from PIL import ImageTk, Image
from OCR import ocrTextDetect
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint
import numpy as np
import loadData
import tokenPadding
import wordEmbedding
import predict
import predictWithSentence
BATCH_SIZE = 25
EPOCHS = 50
LSTM_NODES =256
NUM_SENTENCES = 5390#5000
MAX_SENTENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100
checkpoint_path ="train-weight.ckpt" #"train-wight.ckpt"
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
model.load_weights(checkpoint_path)
senc="after"
print(senc)
output=predictWithSentence.sentence(senc,input_sentences,max_input_len,
             encoder_inputs_placeholder,encoder_states,decoder_embedding,decoder_lstm,
             decoder_dense,word2idx_outputs,MAX_NUM_WORDS,LSTM_NODES,max_out_len)
print(output)


def translate(event,enText,viText):
    enTrans=enText.get("1.0",'end-1c')
    output = predictWithSentence.sentence(enTrans, input_sentences, max_input_len,
                                          encoder_inputs_placeholder, encoder_states, decoder_embedding, decoder_lstm,
                                          decoder_dense, word2idx_outputs, MAX_NUM_WORDS, LSTM_NODES, max_out_len)

    viText.delete("1.0",'end-1c')
    viText.insert("end-1c",output)
def chooseFile(event,self,originalImageLabel,OCRImageLabel,enText,viText):
    filename=filedialog.askopenfile(initialdir = "D:\AI\TranslateMachine\images",
                           title = "Select file",
                           filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    path=filename.name
    print(path)
    self.originalImagePath=path
    self.originalImg = ImageTk.PhotoImage(Image.open(self.originalImagePath).resize((350,200)))
    originalImageLabel.configure(image=self.originalImg)

    senc, pathOCR=ocrTextDetect(self.originalImagePath)
    self.OCRImagePath=pathOCR
    print(pathOCR)
    print(senc)
    self.OCRImg=ImageTk.PhotoImage(Image.open(self.OCRImagePath).resize((350,200)))
    OCRImageLabel.configure(image=self.OCRImg)
    enText.delete("1.0", 'end-1c')
    enText.insert('1.0', senc)
    viText.delete("1.0", 'end-1c')
class Example(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("GROUP 1")

        Style().configure("TButton", padding=(0, 5, 0, 5))

        self.columnconfigure(0, pad=3)
        self.columnconfigure(1, pad=3)
        self.columnconfigure(2, pad=3)


        self.rowconfigure(0, pad=3)
        self.rowconfigure(1, pad=3)
        self.rowconfigure(2, pad=3)
        self.rowconfigure(3, pad=3)
        self.rowconfigure(4, pad=3)
        self.rowconfigure(5, pad=3)
        self.rowconfigure(6, pad=3)
        self.rowconfigure(7, pad=3)
        self.rowconfigure(8, pad=3)

        nameProject=Label(self,text='GROUP 1 : Project Translate Machine',font=("Helvetica", 20),justify='left')
        nameProject.grid(row=0,column=1,rowspan=2,columnspan=2,sticky=E)
        enLabel=Label(self,text="English").grid(row=2,column=0)
        viLabel=Label(self,text='Vietnamese').grid(row=2,column=2)


        enText=Text(self,height=5, width=40)
        enText.grid(row=3,column=0,rowspan=2,columnspan=1)
        viText = Text(self,height=5, width=40)
        viText.grid(row=3, column=2, rowspan=2,columnspan=1)
        translateButton=Button(self,text="TRANSLATE")
        translateButton.grid(row=3,column=1)
        translateButton.bind('<Button-1>',lambda event,arg=(enText,viText):translate(event,enText,viText))



        fileButton=Button(self,text="Choose Image")
        fileButton.grid(row=4,column=1)


        self.iconPath='D:\AI\TranslateMachine\images\\icon.png'
        self.iconImg = ImageTk.PhotoImage(Image.open(self.iconPath).resize((70, 70)))
        iconImageLabel = Label(self, image=self.iconImg)
        iconImageLabel.grid(row=0, column=0,sticky=W+E)

        self.originalImagePath = ''
        originalImageLabel=Label(self)
        originalImageLabel.grid(row=7,column=0,columnspan=1,sticky=W)

        self.OCRImagePath = ''
        OCRImageLabel = Label(self)
        OCRImageLabel.grid(row=7, column=2, columnspan=1, sticky=W)
        enTextLabel = Label(self, text="Original Image",font='Helvetica 18 bold').grid(row=6, column=0)
        viTextLabel = Label(self, text='OCR Image',font='Helvetica 18 bold').grid(row=6, column=2)

        fileButton.bind('<Button-1>', lambda event,
                                             arg=(originalImageLabel,OCRImageLabel,enText,viText): chooseFile(event,self, originalImageLabel,OCRImageLabel,enText,viText))
        self.pack()


root = Tk()
root.geometry("850x550+300+300")
app = Example(root)
root.mainloop()