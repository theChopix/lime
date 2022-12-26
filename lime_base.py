import numpy as np
import itertools

class LimeBase(object):
    """
    LIME generally
    """
    def __init__(self):

    def explain_instance_with_data(self, W, C, p):
        """
        Takes perturbed data, labels and distances, returns explanation

        x_opt = (C.T * diag(W) * C)**(-1) * A.T * C * p

        :param: W - weights (d,) np.array {float}
        :param: C - coalition vectors (d, num_features) np.array {0,1}
        :param: p - predictions of classifier to give data samples (d,) np.array {float}

        :return: x - optimal values - (num_features) np.array {float}
        """
        diag_W = np.diag(W)
        # TODO


def generate_coalition_vectors(num_features):
    """
    Generate perturbed data in binary form (coalitions vectors - matrix)

    :param: num_features - number of features of data samples {int}

    :return: coalitions - coalition vectors in matrix (d, num_features) {0,1}
    """

    coalitions = np.array([list(i) for i in itertools.product([0, 1], repeat=num_features)])
    return coalitions