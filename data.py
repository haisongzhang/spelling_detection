# -*- coding: utf-8 -*- 

import os
import sys
import random
import numpy as np
import xml.dom.minidom
import jieba
import pickle
from pyltp import Segmentor
from pyltp import Postagger

seg = Segmentor()
seg.load('./ltp_data/cws.model')
postag = Postagger()
postag.load('./ltp_data/pos.model')

def get_features(sentence):

    # Filter sentence
    sentence = sentence.replace(' ','')
    # Char
    char = list(sentence)
    word = seg.segment(sentence)
    pos = postag.postag(word)
    # Pos
    pos_ = []
    for i in range(len(word)):
        for j in range(len(word[i])):
                pos_.append(('B-' if j == 0 else 'I-')+pos[i])
    return char, pos_

def train_dataset_loader(file_name):
    with open(file_name, 'r') as my_file:
        DOMTree = xml.dom.minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')
    char_list = []
    tags_list = []
    pos_list = []
    p = 0
    for doc in docs:
        p = p+1
        if p%10 == 0:
            f = open('tags.pkl', 'ab')
            pickle.dump(tags_list, f)
            f.close()
            f = open('pos.pkl', 'ab')
            pickle.dump(pos_list, f)
            f.close()
            f = open('chars.pkl', 'ab')
            pickle.dump(char_list, f)
            f.close()
            print(p)
            del char_list[:]
            del pos_list[:]
            del tags_list[:]
            
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '').encode("UTF-8")
        #text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

        errs = doc.getElementsByTagName('ERROR')
        #corr = doc.getElementsByTagName('CORRECTION')[0].childNodes[0].nodeValue.replace('\n', '')
        char, pos = get_features(text)
        tags = ['O'] * len(char)
        for err in errs:
            start_off = int(err.getAttribute('start_off'))
            end_off = int(err.getAttribute('end_off'))
            err_type = err.getAttribute('type').encode('UTF-8')
            tags[start_off-1] = 'B-' + err_type
            for i in range(start_off, end_off):
                tags[i] = 'I-' + err_type
        char_list.append(char)
        pos_list.append(pos)
        tags_list.append(tags)


if __name__ == '__main__':
    train_file = os.path.join('DATA', 'CGED16_HSK_Train_All.txt')
    train_dataset_loader(train_file)