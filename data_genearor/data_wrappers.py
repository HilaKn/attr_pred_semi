from abc import ABCMeta, abstractproperty
import gzip
import os
import re

from data_genearor.sentence_data import AdjNounData
from parser_wrapper import ParserOutputWrapper as pw


class DataTypes:
    WIKIPEDIA = "wiki"
    GOOGLE_N_GRAMS = "google"
    SYNTACTIC_N_GRAMS = "syntactic"


class DataWrapper(object):

    __metaclass__ = ABCMeta

    def __init__(self, input_path):
        self.input_path = input_path

    @abstractproperty
    def data_collection(self):
        pass


class SyntacticNgrams(DataWrapper):

    FILES_PREFIX = "arcs"

    def __init__(self, input_folder):
        super(SyntacticNgrams,self).__init__(input_folder)
        files = [file for file in  os.listdir(self.input_path) if file.endswith(".gz") and file.startswith(self.FILES_PREFIX)]
        self.ngrams_files = [os.path.join(self.input_path, file) for file in  files]
        print "Total {} files in {}".format(len(self.ngrams_files), self.FILES_PREFIX)


    def get_sentence_data(self, word_data):
        '''
        cease/VB/ccomp/0
        '''
        data = word_data.split('/')
        data_size = len(data)
        word = data[0].lower()
        pos = data[1]
        head = int(data[data_size-1])-1

        return (word,pos,head)

    @property
    def data_collection(self):
        file_number = 0
        for file in self.ngrams_files:
            print "File {} - start iterating over file = {}".format(file_number,os.path.basename(file))
            file_number += 1
            with gzip.open(file, 'rb') as f:
                for line in f:
                    try:
                        adj_noun, skip_flag = self.get_adj_noun(line)
                        if skip_flag:
                            continue
                        yield adj_noun

                    except Exception as e:
                        print "error while iterating line: {}".format(line)
                        print "Error: {}".format(e)

    def __has_digit(self,in_str):
        has_digit = bool(re.search(r'\d', in_str))
        return has_digit

    def get_adj_noun(self, line):
        skip_flag = True
        adj_noun = ""
        line_data = line.split("\t")
        syntactic_data = line_data[1].split()  # this is where the parsing data
        syntactic_count = int(line_data[2])  # total count of this syntactic ngram
        sentence = [self.get_sentence_data(word_data) for word_data in syntactic_data]
        if len(sentence )==2 \
            and len([word_data for word_data in sentence if word_data[1] == pw.ADJ_TAG] ) ==1\
            and len([word_data for word_data in sentence if word_data[1] in pw.NOUN_TAGS] ) ==1:
            adj_noun = AdjNounData(sentence, syntactic_count)
            if self.__has_digit(adj_noun.adj) \
                    or self.__has_digit(adj_noun.noun) \
                    or len(adj_noun.adj) == 1\
                    or len(adj_noun.noun) ==1:
                skip_flag = True
            else:
                skip_flag = False
        return adj_noun, skip_flag

    def get_data_from_single_file(self,file):
        with gzip.open(file, 'rb') as f:
            for line in f:
                try:
                    adj_noun, skip_flag = self.get_adj_noun(line)
                    if skip_flag:
                        continue
                    yield adj_noun
                except Exception as e:
                    print "error while iterating line: {}".format(line)
                    print "Error: {}".format(e)




