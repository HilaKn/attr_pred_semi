import argparse
import time
from data_handler import DataHandler
from predictor import PredictorRunner

def run():
    data_handler = DataHandler()
    runner = PredictorRunner(data_handler, args.model_path, args.train_mode)
    runner.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='semi-supervised attribute predictor')

    parser.add_argument('model_path',help='path for the model pickle - in case of training, this is an output path. in case of testing its from where to load the model')
    parser.add_argument('-t','--train_mode',  default=False, action='store_true',help='train mode or only test if False')

    args = parser.parse_args()
    run()