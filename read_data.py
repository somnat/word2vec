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
    
    def initTableDiscards(self):
        '''
        read section sampling rate
        it describes an equation for calculating a probability with which to 
        keep a given word in the vocabulary. In general, the equation tells us that 
        Smaller values of ‘sample’ means that less likely words are to be kept.
        http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        '''
        sample = 0.001
        z_w = np.array(list(self.word_frequency.values())) / self.tokens
        self.discards = (np.sqrt(z_w / sample) + 1) * (sample / z_w)

    def initTableNegative(self):
        '''
        read section negative sampling and selecting negative samples
        1. The “negative samples” (that is, the 5 output words that we’ll train to output 0) 
        are selected using a “unigram distribution”, where more frequent words are more 
        likely to be selected as negative samples.

        2. The authors state in their paper that they tried a number of variations on this equation, 
        and the one which performed best was to raise the word counts to the 3/4 power

        3. If you play with some sample values, you’ll find that, compared to the simpler equation, this one has the 
        tendency to increase the probability for less frequent words and decrease the probability 
        for more frequent words.

        http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        https://ruder.io/word-embeddings-softmax/
        '''
        power_frequency = np.array(list(self.word.frequency.values())) ** (0.75)
        denominator = sum(power_frequency)
        probability_list = power_frequency / denominator
        count = np.round(probability_list * DataReader.NEGATIVE_TABLE_SIZE)

        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):
        response = self.negatives[self.negpos : self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negpos)

        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response  


        
