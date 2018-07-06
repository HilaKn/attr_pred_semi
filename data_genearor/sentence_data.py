from data_genearor.parser_wrapper import ParserOutputWrapper as pw
class WordData(object):

    def __init__(self,word,pos,head="<NO_NOUN>"):
        self.word = word
        self.pos = pos
        self.head = head

class AdjNounData(object):

    def __init__(self,sentence_data, occurrences=1):

        self.__words_data_sequence = [WordData(word_data[0],word_data[1],word_data[2]) for word_data in sentence_data]
        words_sequence= [word_data.word for word_data in self.__words_data_sequence]
        if self.__words_data_sequence[0].pos == pw.ADJ_TAG:
            self.adj = self.__words_data_sequence[0].word
            self.noun = self.__words_data_sequence[1].word#we nly iterate over pairs of adj and noun so if 0 is adj 1 is noun
        else:
            self.adj = self.__words_data_sequence[1].word
            self.noun = self.__words_data_sequence[0].word


        # self.words_sequence = words_sequence
        # self.pos_sequence =  [word_data.pos for word_data in self.__words_data_sequence]
        # self.head_id_seq = [word_data.head for word_data in self.__words_data_sequence]
        self.sentence_str = "\t".join([self.adj, self.noun])
        self.occurrences = occurrences


    # @property
    # def len(self):
    #     return len(self.__words_data_sequence)
    #
    # def __iter__(self):
    #     return iter(self.__words_data_sequence)
    #
    # def __getitem__(self, key):
    #     return self.__words_data_sequence[key]