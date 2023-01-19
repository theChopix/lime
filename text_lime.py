from lime_base import *
from weight import Weight


class LimeTextExplainer(object):
    """
    LIME for text classification - arrangements for LIME generally (lime_base.py)
    """
    def __init__(self, path_to_instance, classifier, words_to_remove, average_classif_value):
        """
        Class takes testing data and classifier used in predictions
        in the future it may be just training data and their labels insted of classifier

        :param path_to_instance: string
        :param classifier: function
        :param words_to_remove: list
        :param average_classif_value: float
        """
        self.path_to_instance = path_to_instance
        self.classifier = classifier
        self.words_to_remove = np.array([word[0] for word in words_to_remove])
        self.average_classif_value = average_classif_value

    def explain_instance(self, weight_type="shapley"):
        """
        Creates explanation based on class variables (specific data instance, specific classifier,
        words that will be removed, ...) and chosen weight type (default shapley)
         - Computes matrices needed to calculate the values using LIME

        :param weight_type: "shapley" / "banzhaf" {string}

        :return explanation: list of numbers belonging to values
        """

        # coalition vectors
        coalition_vectors = generate_coalition_vectors(len(self.words_to_remove))

        # weight vector
        weight = Weight(coalition_vectors)
        weight_vector = None
        if weight_type == "shapley":
            weight_vector = weight.shapley()
        else:
            weight_vector = weight.banzhaf()

        # predictions vector
        coalitions_num = len(coalition_vectors)
        classifications = np.array([-self.average_classif_value] * coalitions_num)
        for i in range(coalitions_num):
            coalition_indices = np.array(list(map(bool, [not elem for elem in coalition_vectors[i]])))
            w_to_remove = self.words_to_remove[coalition_indices]

            (spam_prob, _), _ = self.classifier.test_instance(self.path_to_instance, words_to_remove=w_to_remove)
            classifications[i] += spam_prob

        return explain_instance_with_data(weight_vector, coalition_vectors, classifications)






