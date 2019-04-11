#coding=utf-8
from . import crnn
import torch
from . import dataset
from torch.autograd import Variable
from PIL import Image
from . import utils
import os
import time

class CharRecognize:
    def __init__(self,weightfile,gpu_id=0):
        alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-()图'
        print(alphabet)
        print(len(alphabet))

        nclass = len(alphabet)+1
        self.__net = crnn.CRNN(32,1,nclass,256)
        if torch.cuda.is_available():
            self.__net.cuda(device=gpu_id)
            self.__gpu_id = gpu_id

        self.__net.load_state_dict(torch.load(weightfile))
        self.__transformer = dataset.resizeNormalize((160,32))
        self.__converter = utils.strLabelConverter(alphabet)

    def recognize(self,im):
        im_gray = im.convert("L")
        img = self.__transformer(im_gray)
        if torch.cuda.is_available():
            img = img.cuda(device=self.__gpu_id)
        img = img.view(1, *img.size())
        img = Variable(img)
        self.__net.eval()
        preds = self.__net(img)
        _,preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = self.__converter.decode(preds.data, preds_size.data, raw=False)

        return sim_pred

def run_0():
    model_path = './models/crnn.pth'
    recognizer = CharRecognize(model_path)
    img_path = './test_images/test2.jpg'
    picdir = "./test_images/"

    for picname in os.listdir(picdir):
        img_path = picdir+picname
        im = Image.open(img_path)

        start = time.time()
        result = recognizer.recognize(im)
        end = time.time()
        print (result,"*******************",(end-start)*1000)

if __name__=="__main__":
    run_0()




