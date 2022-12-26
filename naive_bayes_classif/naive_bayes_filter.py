# from utils import *
# from corpus import Corpus
import numpy as np
import pprint

from naive_bayes_classif.corpus import Corpus
from naive_bayes_classif.utils import read_classification_from_file, SPAM_TAG, HAM_TAG

# indexes in frequency array
SPAM_INDEX = 0
HAM_INDEX = 1


# takes a pattern (regular expression) that is applied to determine from the training data-set
#   characteristics of spam-emails based on the pattern (it's matches)
class NaiveBayesFilter:

    def __init__(self, pattern):
        self.pattern = pattern

        self.is_trained = False

        # words / samples
        self.w = set()

        # P( w | SPAM )
        self.p_w_spam = {}

        # P( w | HAM )
        self.p_w_ham = {}

    def train(self, truth_dict, corpus):
        # N( w | SPAM )
        n_w_spam = {}

        # N( w | HAM )
        n_w_ham = {}

        # N( SPAM )
        n_spam = 0

        # N( HAM )
        n_ham = 0

        for filename, email_body in corpus.emails():
            # filename_samples[filename] = set()

            matches = set(self.pattern.findall(email_body))

            for match in matches:

                self.w.add(match)

                if truth_dict[filename] == 'SPAM':
                    n_w_spam[match] = n_w_spam.get(match, 0) + 1

                elif truth_dict[filename] == 'OK':
                    n_w_ham[match] = n_w_ham.get(match, 0) + 1

            if truth_dict[filename] == 'SPAM':
                n_spam += 1

            elif truth_dict[filename] == 'OK':
                n_ham += 1

        for word in self.w:
            self.p_w_spam[word] = 100 * ( n_w_spam.get(word, 0) + 1 ) / ( n_spam + 1 )
            self.p_w_ham[word] = 100 * ( n_w_ham.get(word, 0) + 1 ) / ( n_ham + 1 )

        self.is_trained = True

    def classify(self, email_body, options=None, n_words=10):
        if self.is_trained:

            spam_prob = 1.
            ham_prob = 1.

            probs = options == "probs"

            main_w = {}

            for word in self.w:
                if word in email_body:
                    spam_prob *= self.p_w_spam[word]
                    ham_prob *= self.p_w_ham[word]

                    if probs:
                        # main_w[word] = spam_prob / (spam_prob + ham_prob)
                        main_w[word] = self.p_w_spam[word] / ( self.p_w_spam[word] + self.p_w_ham[word] )

            spam_prob_norm = spam_prob / (spam_prob + ham_prob)
            ham_prob_norm = 1 - spam_prob_norm

            if probs:
                most_relevant_words = sorted(main_w.items(), key=lambda t: t[1], reverse=True)[0:n_words]
                # pprint.pprint(most_relevant_words)
                # print("spam: " + str(spam_prob_norm) + "; ham: " + str(ham_prob_norm) + "\n")
                return (spam_prob_norm, ham_prob_norm), most_relevant_words
            else:
                return SPAM_TAG if spam_prob_norm > 0.5 else HAM_TAG
        else:
            # in case of non-pretrained classification there is place
            #   for some heuristics, anyway I decided to stick with train technique with given data
            my_truth_dict = read_classification_from_file('spam-data/1/!truth.txt')
            corpus = Corpus("spam-data/1")

            self.train(my_truth_dict, corpus)
            return self.classify(email_body)




