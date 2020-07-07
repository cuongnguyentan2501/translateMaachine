import pytesseract
from pytesseract import  Output
import cv2
import random
import string
def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
def ocrTextDetect(img):
    img = cv2.imread(img, 0)
    custom_config = r'--oem 3 --psm 6'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    d=pytesseract.image_to_data(img, config=custom_config,output_type=Output.DICT)
    # print(d['level'])
    # print(d['conf'])
    # print(d['text'])
    senc=''
    for value in d['text']:
        if(value=='' or value==' '):
            continue
        else:
            senc+=value+' '

    for i in range(len(d['conf'])):
        if(int(d['conf'][i])>40):
            cv2.rectangle(img,(d['left'][i],d['top'][i]),(d['left'][i]+d['width'][i],d['top'][i]+d['height'][i]),(0,0,255),1)
    pathOCR='D:/AI/TranslateMachine/OCRImage/'+str(randomString(8))+'.jpg'

    cv2.imwrite(pathOCR,img)

    return senc,pathOCR

