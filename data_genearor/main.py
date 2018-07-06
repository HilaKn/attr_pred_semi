"""
Generate data set of (adj-noun-freq) for auto-encoder trainer
The triplets should be extracted from the syntactic dependency n-grams
more specifically - use the arcs dataset which contains relations between 2 content words.
we will use adj and noun pos taggs.
Most of the code is similar to the pattern extraction I performed in the past to extracted the
samples from symmetric patterns (adj-noun-adj)
Which is under /home/h/Documents/Hila/Research/code/symetric_patterns_for_adj/patterns_code
"""

import argparse

from data_genearor.adj_noun_extractor import *


def run():

    data_handler = DataHandler( args.input_path)

    if args.parallel:
        data_handler.extract_adj_noun_async()
    else:
        data_handler.extract_adj_noun()

    data_handler.export_results(args.output_folder)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adjectives with multi senses list.')

    parser.add_argument('input_path',help='large text file with POS tagging or folder with sub files')
    parser.add_argument('output_folder', help = 'output folder for the adjectives-nouns pairs')
    parser.add_argument('-p', '--parallel',default=False, action='store_true', help = 'if true run on multi processors')

    args = parser.parse_args()
    start_time = time.time()
    run()

    print ' total running time: {}'.format( time.time() - start_time)
    print "DONE"

