# from naive_bayes_filter import NaiveBayesFilter
# from corpus import Corpus
# from utils import *
import re

from naive_bayes_classif.corpus import Corpus
from naive_bayes_classif.naive_bayes_filter import NaiveBayesFilter
from naive_bayes_classif.utils import read_classification_from_file

lowerPattern = '[a-z]{7,}'


class Filter:

    def __init__(self):
        self.lowerFilter = NaiveBayesFilter(re.compile(lowerPattern))

    # trains each filter instance in self-attributes
    def train(self, train_dir):
        truth_dict = read_classification_from_file(train_dir + '/!truth.txt')

        corpus = Corpus(train_dir)

        self.lowerFilter.train(truth_dict, corpus)

    def test(self, test_dir):
        corpus = Corpus(test_dir)
        for filename, body in corpus.emails():
            # print(filename)
            result = self.lowerFilter.classify(body)
            # file with classification based on the strategy is stored in '!prediction.txt'
            #   in concerned folder (in spam-data..)
            with open(test_dir + "/!prediction.txt", "a+", encoding="utf-8") as prediction:
                prediction.write(filename + " " + result + "\n")

    def test_instance(self, mail_dir, n_words=10):
        with open(mail_dir, "r", encoding='utf-8') as file:
            content = file.read().lower()
            return self.lowerFilter.classify(content, options="probs", n_words=n_words)


