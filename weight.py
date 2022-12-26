import numpy as np
import math


class Weight(object):
    def __init__(self, coalitions):
        self.coalitions = coalitions

    def shapley(self):
        """
        Shapley value

        x(z') = ( M - 1 ) / ( comb(M,|z'|) * |z'| * (M - |z'|) )

        odhad iteracni metodou


        :return: w - weights / distances belonging to perturbed data (d,) d - number of data samples {float}
        """
        M = self.coalitions.shape[1]

        def shapley_value(coalition):
            z = np.count_nonzero(coalition == 1)
            return (M - 1) / ( math.comb(M, z) * z * (M - z) )

        w = np.array([shapley_value(x) for x in self.coalitions])
        return w

    def banzhaf(self):
        """
        Banzhaf value

        :return: w - weights / distances belonging to perturbed data (d,) d - number of data samples {float}
        """
        w = np.array([1.0] * self.coalitions[0])
        return w


if __name__ == "__main__":
    a = np.array([[0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 0],
                  [1, 1, 0, 1, 1],
                  [1, 0, 1, 0, 0]])
    weights = Weight(a)
    banzhaf_weights = weights.banzhaf()
    shapley_weights = weights.shapley()
    print(bahnzaf_weights)
    print(shapley_weights)

