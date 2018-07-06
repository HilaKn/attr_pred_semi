"""
Split the adj-noun-count data set into 3 separated ds
"""
import numpy as np
import argparse
from random import shuffle
import time
from adj_noun_auto_encoder.we_wrapper import we_model
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 412


def generate_output(data, suffix):
    file_path = "../dataset/adj_noun_{}".format(suffix)
    with open(file_path, "w") as f:
        for samp in data:
            f.write(samp + "\n")
    print("done saving {} samples. total {}".format(suffix, len(data)))

def verify_sample(row_data):
    flag = False
    data = row_data.split("\t")
    adj = data[0]
    noun = data[1]
    if adj in we_model and noun in we_model:
        flag = True
    return flag

def run():

    with open(args.input_path) as f:
        lines = f.readlines()
        dataset = [line.rstrip('\r\n') for line in lines]
        dataset = [samp for samp in dataset if verify_sample(samp)]

    shuffle(dataset)

    dataset_size = len(dataset)
    test_split = int(np.floor(TEST_SPLIT * dataset_size))
    validation_split = int(np.floor(VALIDATION_SPLIT * dataset_size))

    train =  dataset[validation_split + test_split:]
    generate_output(train, "train")

    val = dataset[:validation_split]
    generate_output(val, "val")

    test= dataset[validation_split:validation_split + test_split]
    generate_output(test, "test")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adjectives with multi senses list.')

    parser.add_argument('input_path',help='input file path containing adj-noun-count')


    args = parser.parse_args()
    start_time = time.time()
    run()

    print ' total running time: {} sec'.format( time.time() - start_time)
    print "DONE"

