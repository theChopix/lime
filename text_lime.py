from lime_base import LimeBase
from lime_base import generate_coalition_vectors


class LimeTextExplainer(object):
    """
    LIME for text classification - arrangements for LIME generally (lime_base.py)
    """
    def __init__(self, data, classifier):
        """
        Class takes testing data and classifier used in predictions
        in the future it may be just training data and their labels insted of classifier

        :param data:
        :param classifier:
        """
        self.data = data
        self.classifier = classifier

    def features_selection(self, num_features):
        """
        Selecting words in data instance that will be used in explanation
            - most frequent words in training data-set ?

        selekce na zaklade klasifikatoru z RPZ

        :param: data instance
        :param: data

        :return:
        """
        # TODO

    def explain_instance(self, num_features, weight=None):
        """
        :param: num_features - number of features (words) chosen in features selection int
        :param: weight - "shapley" / "banzhaf" {string}
        :param: data
        :param: classifier - classifier {function}
        """

        coalition_vectors = generate_coalition_vectors(num_features)

        if weight is None:
            weight = "shapley"

        # TODO

