from pred_models import *
from config import *
from torch.autograd import Variable
import torch
import numpy as np
import random
import operator
from scipy import spatial
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from data_handler import AdjNounAttribute
from we_wrapper import we_model
from sklearn import preprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from logger import logger
import copy
import time



class PredictorRunner(object):

    def __init__(self, data_handler, model_path, train_flag=True):
        self.__data_handler = data_handler
        self.cuda = CUDA_FLAG and torch.cuda.is_available()
        logger.info("Using cuda: {}".format(self.cuda))
        self.nn_model = model_factory(MODEL_TYPE)
        if  self.cuda:
            self.nn_model.cuda()
        self.criterion = torch.nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(),lr=LR)#torch.optim.Adam(self.nn_model.parameters(), lr=LR)
        self.train_flag = train_flag
        self.model_path = model_path



    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = LR * (ADJUST_LR_FACTOR ** (epoch // ADJUST_LR_K_EPOCHS))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def __train_model(self):
        since = time.time()
        torch.set_num_threads(THREADS)
        best_model_wts = copy.deepcopy(self.nn_model.state_dict())
        best_acc = 0.0

        for epoch in xrange(0, EPOCHS):
            logger.info("Epoch: {}\{}".format(epoch, EPOCHS-1))
            logger.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                running_loss = 0.0
                if phase == 'train':
                    self.adjust_learning_rate(epoch)
                    self.nn_model.train()  # Set model to training mode
                else:
                    self.nn_model.eval()   # Set model to evaluate mode

                for batch_idx, (x_train, y_train, has_label) in enumerate(self.__data_handler.train_loader):

                    if self.cuda:
                        x, y = Variable(x_train.cuda()), Variable(y_train.cuda())
                        with_label_indces = torch.LongTensor([idx for idx,item in enumerate(has_label) if item]).cuda()
                        # with_label_indces = torch.tensor([0, 2])
                    else:
                        x, y = Variable(torch.Tensor(x_train)), Variable(torch.Tensor(y_train))
                        with_label_indces = torch.LongTensor([idx for idx,item in enumerate(has_label) if item])
                        # with_label_indces = torch.tensor([0, 2])

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = self.nn_model(x)


                        y_pred_ae = y_pred[:,0:2*WE_DIM] #only the auto-encoder part - adj-noun
                        y_pred_attr = torch.index_select(y_pred, 0, with_label_indces)[:,2*WE_DIM:]# y_pred[with_label_indces:,2*WE_DIM:] #the attr part only for samples with label



                        y_ae = y[:,0:2*WE_DIM]
                        y_attr = torch.index_select(y, 0, with_label_indces)[:,2*WE_DIM:] #y[with_label_indces:,2*WE_DIM:]

                        if self.cuda:
                            y_pred_ae = y_pred_ae.cuda()
                            y_pred_attr = y_pred_attr.cuda()
                            y_ae = y_ae.cuda()
                            y_attr = y_attr.cuda()

                        auto_encoder_loss = self.criterion(y_pred_ae, y_ae)
                        # logger.info("first part loss")
                        # if len(with_label_indces) > 0:
                        #     with_labels_loss = self.criterion(y_pred_attr, y_attr)
                        # else:
                        #     y_attr = y_pred_attr.clone()
                        #     with_labels_loss = self.criterion(y_pred_attr, y_attr)#use 0 loss in case of no label
                        #     with_labels_loss = torch.zeros()
                        # logger.info("seconf part loss")
                        loss = auto_encoder_loss# + LABEL_LOSS_WEIGHT *with_labels_loss
                        if len(with_label_indces) > 0:
                            # logger.info("add the label part loss")
                            with_labels_loss = self.criterion(y_pred_attr, y_attr)
                            weight = LABEL_LOSS_WEIGHT*len(x)/len(with_label_indces)#
                            loss += weight *with_labels_loss
                        # logger.info("full loss")

                        #backward + optimize only for training phase
                        if phase == 'train':
                            loss.backward()
                            # logger.info("after backward")
                            self.optimizer.step()



                    # statistics
                    running_loss += loss.item() * x.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                    if (batch_idx) % LOG_INTERVAL == 0 and batch_idx>0:
                        samp_in_ep_so_far = max(batch_idx * len(x),1.0)
                        logger.info('{} Epoch: {} Samples:{}\tTotal loss: {:.6f} Loss: {:.6f}'.format(phase,
                        epoch, samp_in_ep_so_far, running_loss,
                        running_loss/samp_in_ep_so_far))



                epoch_loss = running_loss / self.__data_handler.ds_sizes[phase]
                epoch_acc = self.eval_model_accuracy(phase)
                logger.info('{} Loss: {:.5f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.nn_model.state_dict())

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        # logger.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.nn_model.load_state_dict(best_model_wts)

    def calc_mat_distance(self,x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, 2).sum(2)
        return dist


    def eval_model_accuracy(self, phase, export_res_to_file=False):
        logger.info("eval_model_accuracy for phase: [{}]".format(phase))
        self.nn_model.eval()
        correct = 0.0
        results = []
        org_input = []
        if phase == "train":
            data_handler = self.__data_handler.heiplas_train
        elif phase == "val":#validation
            data_handler = self.__data_handler.heiplas_val
        elif phase == "test":
            data_handler = self.__data_handler.heiplas_test
        else:
            raise Exception("can't get data handler for phase:{}".format(phase))

        for batch_idx, (x_train, y_train, samples) in enumerate(data_handler):
            if self.cuda:
                data, target = x_train.cuda(), y_train.cuda()
                attributes = data_handler.dataset.attr_matrix.cuda()
            else:
                data, target = torch.Tensor(x_train), torch.Tensor(y_train)
                attributes = torch.Tensor(data_handler.dataset.attr_matrix)
            data, target, attributes = Variable(data), Variable(y_train),Variable(attributes)

            with torch.set_grad_enabled(False):
                y_pred = self.nn_model(data)
            pred_vectors = y_pred[:,WE_DIM*2:]#the last WE_DIM entries are the predicted attribute vector
            if self.cuda:
                pred_vectors = pred_vectors.cuda()

            dist = self.calc_mat_distance(pred_vectors, attributes )
            values, indices = torch.max(dist, 1)#in case of similarity this should be changed to min
            pred_attributes = [data_handler.dataset.unique_attributes[i] for i in list(indices)]

            correct+= len([1 for pred_attr, sample in zip(pred_attributes, samples)
                        if pred_attr.lower() == sample.split()[0].lower()])
            org_input.extend(samples)
            results.extend(pred_attributes)

        accuracy = float(correct)/len(data_handler.sampler)


        logger.info( "phase:{} correct = {}, total: {}, accuracy: {}".format(phase,correct, len(data_handler.sampler), accuracy))

        if export_res_to_file:
            output_file = "{}_{}".format(RESULTS_FILE,phase)
            logger.info("export results to file: {}".format(output_file))
            with open(output_file,'w') as file:
                for input,pred in zip(org_input,results):
                    string = ' '.join([str(input),pred.upper()])
                    # print string
                    print >>file,string
        return accuracy



    def __save_model(self):
        print "Start saving model to: {}".format(self.model_path)
        torch.save(self.nn_model.state_dict(), self.model_path)
        print "Done saving model to pickle file"


    def __load_model(self):
        print "Start loading model from: {}".format(self.model_path)
        self.nn_model.load_state_dict(torch.load(self.model_path))
        print "Done loading adj prediction model"




    def run(self):
        if self.train_flag:
            self.__train_model()
            self.__save_model()
        else:
            self.__load_model()

        self.eval_model_accuracy("test",export_res_to_file=True)
        #TODO:consider implementing regular test that will measure only loss an not accuracy
       # self.__test()




