import torch
import numpy as np
from torch.utils.data import Dataset

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count):
        self.negatives = []
        self.discards = []
        self.negpos = 0
        
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_freq = dict()
        sentences = open(self.inputFileName, encoding='utf-8', mode='r').readlines()
        self.sentences_count = sum(
            [1 for sentence in sentences if len(sentence.strip()) > 1]
        )
        for sentence in sentences:
             for word in sentence.strip():
                 if len(word) > 0:
                     word_freq[word] += 1
                     self.token_count += 1
        wid = 0
        for word, count in word_freq.items():
            if count < min_count:
                continue
            self.word2id[word] = wid
            self.id2word[wid] = word
            self.word_frequency[wid] = count
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))
    
    
        
