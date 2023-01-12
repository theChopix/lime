from naive_bayes_classif.corpus import Corpus
from naive_bayes_classif.utils import read_classification_from_file, SPAM_TAG, HAM_TAG

SPAM_INDEX = 0
HAM_INDEX = 1


# takes a pattern (regular expression) that is applied to determine from the training data-set
#   characteristics of spam-emails based on the pattern (it's matches)
class NaiveBayesFilter:
    """
    Naive Bayesian Filter following classic bayesian formulas
    """

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
        """
        Training Naive Bayesian Filter
        - Computing prior probabilities of each word in each email (in directory)
         based on their occurrences in SPAM/HAM emails

        :param truth_dict: ground truth classifications
        :param corpus: generator (yield) of training emails in directory
        """

        # N( w | SPAM )
        n_w_spam = {}

        # N( w | HAM )
        n_w_ham = {}

        # N( SPAM )
        n_spam = 0

        # N( HAM )
        n_ham = 0

        for filename, email_body in corpus.emails():

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

    def classify(self, email_body, num_words=0):
        """
        Classifying single data-instance (based on its content - email_body)
        - Aggregating (multiplying) prior probabilities computed in train and returning resulting values

        :param email_body: string content of examined email
        :param num_words: optional parameter indicating how much 'most relevant words' we want to get from function
            according to its interpretation

        :return: (spam_prob, ham_prob), most_relevant_words
         * most_relevant_words -> empty list in case that num_words is 0
        """

        if self.is_trained:

            spam_prob = 1.
            ham_prob = 1.

            main_words = {}

            for word in self.w:
                if word in email_body:
                    spam_prob *= self.p_w_spam[word]
                    ham_prob *= self.p_w_ham[word]

                    if num_words > 0:
                        main_words[word] = self.p_w_spam[word] / ( self.p_w_spam[word] + self.p_w_ham[word] )

            spam_prob_norm = spam_prob / (spam_prob + ham_prob)
            ham_prob_norm = 1 - spam_prob_norm

            most_relevant_words = sorted(main_words.items(), key=lambda t: t[1], reverse=True)[0:num_words]
            return (spam_prob_norm, ham_prob_norm), most_relevant_words

        else:
            # in case of non-pretrained classification there is place
            #   for some heuristics, anyway I decided to stick with train technique with given data
            my_truth_dict = read_classification_from_file('spam-data/1/!truth.txt')
            corpus = Corpus("spam-data/1")

            self.train(my_truth_dict, corpus)
            return self.classify(email_body)




