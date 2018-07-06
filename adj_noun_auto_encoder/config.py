
#we_model
NORMED_MODEL = True #used the normed word embedding model
WORD2VEC_FILE_PATH  = "/home/h/data/word2vec/word2vec_text"#"/home/h/data/word2vec/GoogleNews-vectors-negative300.bin.gz"#

#data_loader
PRINT_EVERY = 100
SAMPLES_ABOVE_THRESHOLD_FLAG = False #indicating if taking samples with more than SAMPLE_COUNT_THRESHOLD count or use all samples
SAMPLE_COUNT_THRESHOLD = 50 #if SAMPLES_ABOVE_THRESHOLD_FLAG is True, this is the minimum count
DEFAULT_SAMPLE_COUNT = 100 #for samples that don't have count (e.g. from HeiPLAS) use this count
SAMPLE_WEIGHT_TYPE = "count"
DATA_LOADER_WORKERS =4
HEIPLAS_VALIDATION_SIZE = 0.5#validation set of the full adj-noun-attr dataset - takem from the heiplas training file
RANDOM_SEED = 247
HEIPLAS_BATCH_SIZE = 30

#sampler
SAMPLE_FACTOR = 1

#train
BATCH_SIZE = 5
LR = 0.01
EPOCHS = 5
ADJUST_LR_K_EPOCHS = 500 #set to more then EPOCHS for no adjusting learning rate
ADJUST_LR_FACTOR = 0.1
LABEL_LOSS_WEIGHT = 1.0 #the loss is combined from loss for labeled and unlabeled samples. this is the labeled weight
LOG_INTERVAL = 100

#test
TEST_BATCH_SIZE = 50

#nn_model
MODEL_TYPE= "MLP"

#MLP params
D_IN = 600
D_OUT = 900
D_HIDDEN = 200
HIDDEN_LAYERS = 1
USE_DROPOUT = False
DROPOUT_RATE = 0.2

#files
ADJ_NOUN_TRAIN_FILE = "../dataset/adj_noun_train"
ADJ_NOUN_VALIDATION_FILE = "../dataset/adj_noun_val"
ADJ_NOUN_TEST_FILE = "../dataset/adj_noun_test"
HEIPLAS_TRAIN_FILE = "/home/h/data/HeiPLAS_filtered/HeiPLAS-dev.txt"
HEIPLAS_TEST_FILE = "/home/h/data/HeiPLAS_filtered/HeiPLAS-test.txt"
RESULTS_FILE = "../results/predictions"


#dimmenssions
WE_DIM = 300

CUDA_FLAG = False
THREADS = 1
DEBUG_MODE = True #in debug mode running only on small sample