import re
import numpy as np

from naive_bayes_classif.corpus import Corpus
from naive_bayes_classif.naive_bayes_filter import NaiveBayesFilter
from naive_bayes_classif.utils import read_classification_from_file, SPAM_TAG, HAM_TAG

# searching for words with at least 7 letters - word-based filter
lowerPattern = '[a-z]{7,}'


class Filter:
    """
    General Filter Class implementing interface between specific filter and data
    """

    def __init__(self):
        self.lowerFilter = NaiveBayesFilter(re.compile(lowerPattern))

    def train(self, train_dir):
        """
        Method for training the filter (on whole data directory)

        :param train_dir: path to directory in which are email-data together with ground-truth file (!truth.txt)
        """

        truth_dict = read_classification_from_file(train_dir + '/!truth.txt')

        corpus = Corpus(train_dir)

        self.lowerFilter.train(truth_dict, corpus)

    def test(self, test_dir):
        """
        Method for testing (classification) the filter (on whole data directory)

        :param test_dir: path to directory in which are email-data
         - eventually the method creates prediction to each email in directory and
         saves it to the classification file (!prediction.txt)

         :return average_spam_value: average spam value of all email classifications
          - used later in explanations
        """

        corpus = Corpus(test_dir)

        prediction_str = str()
        spam_probs = np.array([])

        for filename, body in corpus.emails():

            (spam_prob, ham_prob), _ = self.lowerFilter.classify(body)
            spam_probs = np.append(spam_prob, spam_prob)
            result = SPAM_TAG if spam_prob > ham_prob else HAM_TAG

            prediction_str += filename + " " + result + "\n"

        # filenames with its classifier's predictions are stored in '!prediction.txt'
        #   in concerned folder (in data/..)
        with open(test_dir + "/!prediction.txt", "w", encoding="utf-8") as prediction_file:
            prediction_file.write(prediction_str)

        prediction_file.close()

        # during classification stores information about spam-values of each email and
        #  computes average spam value and returns it - it can be handy during classification explanations
        average_spam_value = np.mean(spam_probs)
        return average_spam_value

    def test_instance(self, mail_dir, n_words=0, words_to_remove=None):
        """
        Method used for testing single data instance (single email)
         - used in interpretations and explanations

        :param mail_dir: path to specific data instance
        :param n_words: optional parameter used in interpretations
            to indicate how much 'most relevant words' we want to get from function
        :param words_to_remove: optional parameter used in explanations
            - words that will be removed from content of data instance before its classification

        :return: classification - (spam_value, ham_value), most_relevant_words
        """

        with open(mail_dir, "r", encoding='utf-8') as file:
            content = file.read().lower()

            if words_to_remove is not None:
                for word in words_to_remove:
                    content = content.replace(word, '')

            classification = self.lowerFilter.classify(content, num_words=n_words)

        file.close()

        return classification


