from collections import defaultdict, Counter
import time

from multiprocess import Value, Pool, cpu_count, Lock

from data_wrappers import *


NO_HEAD_NOUN = "<NO_NOUN>"
WORKERS = cpu_count() - 2
sent_locker = Lock()
lock = Lock()
sentence_counter = Value("i",0)


class Pattern(object):
    POS_TAG = 'JJ'
    POSSIBLE_TAGS = ['JJ','NN', 'NNS', 'NNP', 'NNPS']
    NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']
    ADJ_TAGS = ['JJ']


class DataHandler(object):

    __metaclass__ = ABCMeta

    def __init__(self, input_text_file,read_file_format='rb'):
        self.read_format = read_file_format
        self.adj_noun_to_count = defaultdict(int)
        self.data_wrapper = SyntacticNgrams(input_text_file)

    def extract_adj_noun(self):
        sentence_id = 0
        for adj_noun in self.data_wrapper.data_collection:
            self.adj_noun_to_count[adj_noun.sentence_str] += adj_noun.occurrences
            sentence_id += 1
            # print "sentence {}".format(sentence_id)
            if sentence_id % 100000 == 0:
                print "finished process sentence {}".format(sentence_id)
                # break


    def __extract_patterns_from_file(self, file):
        counter = 0
        adj_noun_to_count = defaultdict(int)
        print "running on file: {}".format(file)
        for adj_noun in self.data_wrapper.get_data_from_single_file(file):
            adj_noun_to_count[adj_noun.sentence_str] += adj_noun.occurrences
            counter +=1
            if counter % 100000 == 0:
                print "finished process sentence {} from file: {}".format(counter, file)
                # break
        return adj_noun_to_count


    def extract_adj_noun_async(self):
        startTime = time.time()
        counters = []
        print "running on {} processors".format(WORKERS)
        pool = Pool(processes=WORKERS)#,initargs=(sent_locker,lock, sentence_counter))
        # adj_noun_dict = pool.map(self.__extract_patterns_from_file, self.data_wrapper.ngrams_files)
        results = [pool.apply_async( self.__extract_patterns_from_file, (file,) ) for file in self.data_wrapper.ngrams_files]

        # for res in results:
        #     dict_res = res.get()
        #     res_counter = Counter(dict_res)
        #     counters.append(res_counter)
        counters = [Counter(x.get()) for x in results]
        print "starting to sum counters from {} dictionaries".format(len(counters))
        self.adj_noun_to_count = sum(counters,Counter())
        print "done summing counters"
        # pool.close()
        # pool.join()
        total_time = time.time()-startTime
        print "extract_adj_noun_async running time: {}".format(total_time)


    def export_results(self,output_folder):
        # localtime = time.asctime( time.localtime(time.time()) )
        timestr = time.strftime("%Y%m%d-%H%M%S")

        output_folder = output_folder+'_'+timestr
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_name = os.path.join(output_folder, "adj_noun_data")

        with open(file_name,'w') as f:
            for adj_noun, count in self.adj_noun_to_count.iteritems():
                row = "{}\t{}\n".format(adj_noun, count)
                f.write(row)




