import gensim
import os
from config import *


class WordVecWrapper(object):

    def __init__(self,we_file):
        self.we_file = we_file
        self.__out_of_vocab = set()

    def set_model(self):
        if not NORMED_MODEL:
            # load pre-trained word2ec
            print "start loading word2vec model from: {}".format(self.we_file)
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.we_file, binary=True)
            # self.model = gensim.models.KeyedVectors.load(self.we_file, mmap='r')  # mmap the large matrix as read-only
            print "done loading word2vec model from"
        else:
            print "start loading normed word2vec model from: {}".format(self.we_file)
            self.model = gensim.models.KeyedVectors.load(self.we_file, mmap='r')  # mmap the large matrix as read-only
            self.model.syn0norm = self.model.syn0
            print "done loading normed word2vec model from"

        print "testing word2vec model"
        print "vocab size = {}".format(len(self.model.vocab))
        word = "hot"
        # print "{} vector = {}".format(word, self.model[word])
        # print "{} most sim: {}".format(word,self.model.similar_by_word(word,5))
        print "done testing model"

    def is_all_in_vocab(self, list_of_words):
        all_in_vocab = True
        for word in list_of_words:
            if word not in self.model.vocab:
                # print "[{}] is not in vocab".format(word)
                self.__out_of_vocab.add(word)
                all_in_vocab = False
                break

        return all_in_vocab

    @property
    def total_out_of_vocab(self):
        return self.__out_of_vocab

we_wrapper = WordVecWrapper(WORD2VEC_FILE_PATH)
we_wrapper.set_model()
we_model = we_wrapper.model

