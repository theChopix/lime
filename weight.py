import numpy as np
import math


class Weight(object):
    def __init__(self, coalitions):
        self.coalitions = coalitions

    def shapley(self):
        """
        Shapley value
        - weights that respect the size of the coalition

        x(z') = ( M - 1 ) / ( comb(M,|z'|) * |z'| * (M - |z'|) )

        :return w: weights / distances belonging to perturbed data (d,) d - number of data samples {float}
        """
        M = self.coalitions.shape[1]

        def shapley_value(coalition):
            z = np.count_nonzero(coalition == 1)
            return (M - 1) / ( math.comb(M, z) * z * (M - z) )

        w = np.array([0.0] * len(self.coalitions))
        for i in range(len(self.coalitions)):

            # for full/empty vector we would get division by zero according to formula above,
            #  therefore we assign constant (1) to its weight
            if np.count_nonzero(self.coalitions[i] == 0) == M or np.count_nonzero(self.coalitions[i] == 1) == M:
                w[i] = 1.
            # in other cases we follow formula above
            else:
                w[i] = shapley_value(self.coalitions[i])

        return w

    def banzhaf(self):
        """
        Banzhaf value
        - weights that assign the same 'value' to each coalition

        :return w: weights / distances belonging to perturbed data (d,) d - number of data samples {float}
        """
        w = np.array([1.0] * self.coalitions.shape[0])
        return w


if __name__ == "__main__":
    a = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [1, 1, 0, 1, 1],
                  [1, 0, 1, 0, 0]])
    weights = Weight(a)
    banzhaf_weights = weights.banzhaf()
    shapley_weights = weights.shapley()
    print(banzhaf_weights)
    print(shapley_weights)

