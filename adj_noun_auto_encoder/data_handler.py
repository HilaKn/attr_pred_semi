import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler, RandomSampler, SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple, defaultdict
from we_wrapper import we_model
import torch
import numpy as np
from collections import namedtuple
import math
from logger import logger
from config import *
from sample_weight_producer import *

# AdjNounAttribute = namedtuple("AdjNounAttribute", 'adj, noun, attr')
class AdjNounAttribute(namedtuple("AdjNounAttribute", ["adj", "noun", "attr"])):

    def __str__(self):
        return " ".join([self.attr.upper(), self.adj, self.noun])
#
# class AdjNounAttribute(object):
#     def __init__(self, adj, noun, attr):
#         self.adj = adj
#         self.noun= noun
#         self.attr = attr
#
#     def __str__(self):
#         return " ".join([self.attr.upper(), self.adj, self.noun])

AdjNoun = namedtuple("AdjNoun", 'adj, noun')


UNKNOWN_WORD = "<UNKNOWN_WORD>"
HAS_ATTR = 1.0
NO_ATTR = 0.0

class WordData(object):
    def __init__(self, word):
        self.__word = word
        # if word != UNKNOWN_WORD:
        #     self.__vec = we_model[word]
        # else:
        #     self.__vec = np.zeros(WE_DIM, dtype=np.float32)

    @property
    def word(self):
        return self.__word

    @property
    def vec(self):
        if self.word != UNKNOWN_WORD:
            return we_model[self.word]
        else:
            return np.zeros(WE_DIM, dtype=np.float32)


    def __hash__(self):
        return hash(self.__word)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.word == other.word

class Sample(object):
    def __init__(self,sample_raw_data, attr=UNKNOWN_WORD):
        '''
        get sample raw data as tuple of (adj,noun,count)
        '''
        self.adj = WordData(sample_raw_data[0])
        self.noun = WordData(sample_raw_data[1])
        self.attr = WordData(attr) #could not be part of the hash cause it might change in time
        self.count = sample_raw_data[2]
        self.weight = get_weight(self.count)
        # self.__x = np.concatenate((self.adj.vec,self.noun.vec))


    def update_attr(self,attr):
        if attr in we_model:
            self.attr = WordData(attr)

    @property
    def has_attr(self):
        if self.attr.word == UNKNOWN_WORD:
            return False
        else:
            return True

    @property
    def x(self):
        return np.concatenate((self.adj.vec,self.noun.vec))

    @property
    def y(self):
        return np.concatenate((self.x, self.attr.vec))

    def __hash__(self):
        return hash((self.adj,self.noun))

    def __eq__(self, other):
        return self.__class__ == other.__class__ \
               and self.adj.word == other.adj.word \
               and self.noun.word == other.noun.word



class AdjNounAttrDs(data.Dataset):
    '''
    data set of adj-noun and attribute if exists. if not use "<UNKNOWN_ATTR>" flag
    '''
    def __init__(self, input_file, heiplas_samples):
        counter = 0
        samples = []

        #load HeiPLAS ds
        adj_noun_to_attr = {}
        attr_set = set()
        for sample  in heiplas_samples:
            adj_noun_to_attr[(sample.adj,sample.noun)] = sample.attr
            attr_set.add(sample.attr)

        #load adj_noun_count samples
        with open(input_file) as f:
            for row in f:
                try:
                    counter+=1
                    if counter % PRINT_EVERY == 0:
                        logger.info("processing {} row for AdjNounAttrDs".format(counter))
                        if DEBUG_MODE:
                            logger.info("debug mode - stop loading data")
                            break
                    row_data = row.rstrip('\n\r').split("\t")
                    sample = Sample(row_data)
                    #if the ample exist in HeiPLAS, so it has attribute - update the attribute
                    if adj_noun_to_attr.has_key((sample.adj.word, sample.noun.word)):
                        logger.debug("updating attribute for ({},{})".format(sample.adj.word, sample.noun.word))
                        sample.update_attr(adj_noun_to_attr[(sample.adj.word, sample.noun.word)])
                        del adj_noun_to_attr[(sample.adj.word, sample.noun.word)]
                    samples.append(sample)
                except:
                    logger.exception("exception while processing row:[{}]".format(row))

        logger.info("Number of samples: [{}]".format(len(samples)))
        logger.info("Going to add {} samples from Heiplas".format(len(adj_noun_to_attr)))

        #iterate over remainig items in heiplas dict to add them to the dataset
        for (adj,noun), attribute in adj_noun_to_attr.iteritems():
            sample = Sample((adj,noun,1),attr=attribute)
            samples.append(sample)
            # logger.debug("adding heiplas sample: {} {} {}".format(adj,noun,attribute.upper()))


        if SAMPLES_ABOVE_THRESHOLD_FLAG:
            logger.info("filtering out samples with count < [{}]".format(SAMPLE_COUNT_THRESHOLD))
            self.__samples = set([sample for sample in samples if sample.count >= SAMPLE_COUNT_THRESHOLD])
        else:
            self.__samples = set(samples)
        logger.info("Number of unique samples : [{}]".format(len(self.__samples)))

        self.__samples = list(self.__samples)

        #updating samples with attribute label to the maximum sampling weight
        max_weight = max([samp.weight for samp in self.__samples])
        for samp in self.__samples:
            if samp.attr.word != UNKNOWN_WORD:
                samp.weight = max_weight


    @property
    def samp_weights(self):
        return [samp.weight for samp in self.__samples]

    def __getitem__(self, index):
        input, target, has_label = self.__samples[index].x, self.__samples[index].y, self.__samples[index].has_attr
        return input, target, has_label

    def __len__(self):
        return len(self.__samples)


class HeiPlasDs(data.Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.dev_data = []
        self.dev_targets = []
        self.dev_adj_noun_attr_list = []
        self.unique_attributes = []
        self.__unique_attr_matrix = []

        with open(self.input_file) as f:
            attr_set = set()
            for row in f:
                attr,adj,noun = row.rstrip("\n").split()
                attr = attr.lower()
                attr_set.add(attr)
                # adj_vec = we_model[adj]
                # noun_vec = we_model[noun]

                self.dev_adj_noun_attr_list.append(AdjNounAttribute(adj,noun,attr))

                # in_dev = np.concatenate((adj_vec,noun_vec))
                # self.dev_data.append(in_dev)
                # self.dev_targets.append(attr)

        attr_set.discard("good")
        self.unique_attributes = list(attr_set)

    @property
    def attr_matrix(self):
        return torch.Tensor([we_model[attr] for attr in self.unique_attributes])

    def __getitem__(self, index):
        anat =  self.dev_adj_noun_attr_list[index]
        input = np.concatenate((we_model[anat.adj] ,we_model[anat.noun] ))
        target = np.array(we_model[anat.attr])
        org_data = str(anat)
        # input, target, org_data = np.array(self.dev_data[index]), np.array(we_model[attr]), str(self.dev_adj_noun_attr_list[index])
        return input, target, org_data

    def __len__(self):
        return len(self.dev_adj_noun_attr_list)



class DataHandler(object):

    def __init__(self):

        heiplas_train_ds = HeiPlasDs(HEIPLAS_TRAIN_FILE)
        num_train = len(heiplas_train_ds)
        indices = list(range(num_train))
        split = int(np.floor(HEIPLAS_VALIDATION_SIZE * num_train))


        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        self.heiplas_train = torch.utils.data.DataLoader(heiplas_train_ds,
                                            batch_size=HEIPLAS_BATCH_SIZE,
                                            sampler=train_sampler,
                                            num_workers=DATA_LOADER_WORKERS)

        self.heiplas_val = torch.utils.data.DataLoader(heiplas_train_ds,
                                            batch_size=HEIPLAS_BATCH_SIZE,
                                            sampler=valid_sampler,
                                            num_workers=DATA_LOADER_WORKERS)

        heiplas_test_ds = HeiPlasDs(HEIPLAS_TEST_FILE)
        test_sampler = SequentialSampler(heiplas_test_ds)
        self.heiplas_test = torch.utils.data.DataLoader(heiplas_test_ds,
                                            batch_size=HEIPLAS_BATCH_SIZE,
                                            sampler=test_sampler,
                                            num_workers=DATA_LOADER_WORKERS)



        self.train_ds = AdjNounAttrDs(ADJ_NOUN_TRAIN_FILE, [heiplas_train_ds.dev_adj_noun_attr_list[idx] for idx in train_idx])
        self.train_sampler = WeightedRandomSampler(weights = self.train_ds.samp_weights,
                                            num_samples = len(self.train_ds)*SAMPLE_FACTOR,
                                            replacement=True)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_ds,
                                           sampler = self.train_sampler,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=DATA_LOADER_WORKERS)


        self.val_ds = AdjNounAttrDs(ADJ_NOUN_VALIDATION_FILE, [heiplas_train_ds.dev_adj_noun_attr_list[idx] for idx in valid_idx])
        self.val_sampler = WeightedRandomSampler(weights = self.val_ds.samp_weights,
                                            num_samples = len(self.val_ds)*SAMPLE_FACTOR,
                                            replacement=True)

        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_ds,
                                           sampler = self.val_sampler,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=DATA_LOADER_WORKERS)

        self.test_ds = AdjNounAttrDs(ADJ_NOUN_TEST_FILE, heiplas_test_ds.dev_adj_noun_attr_list)
        self.test_sampler = SequentialSampler(self.test_ds)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_ds,
                                                       sampler=self.test_sampler,
                                                       batch_size=TEST_BATCH_SIZE,
                                                       shuffle=False)



        self.ds_sizes = {"train":self.train_size, "val":self.val_size, "test": self.test_size}

    @property
    def train_size(self):
        return len(self.train_ds)

    @property
    def val_size(self):
        return len(self.val_ds)

    @property
    def test_size(self):
        return len(self.test_ds)

    def size(self,type):
        if type == "train":
            return






