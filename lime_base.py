import numpy as np
import itertools


def explain_instance_with_data(W, A, b):
    """
    Takes perturbed data, labels and distances, returns explanation

    x_opt = (A.T * diag(W) * A)**(-1) * A.T * C * p

    :param: W - weights, np array (2^n, 1) {float}
    :param: A - coalition vectors, np array (2^n, n) {0,1}
    :param: b - game values, np.array (1, 2^n) {float}

    :return: x - optimal values, np.array (n+1, 1) or (n, 1)
    """

    # making weights diagonal, np array (2^n, 2^n)
    diag_W = np.diag(W)

    # finding values can be described as the problem:
    # min(A@x-b).T@W@(A@x-b)

    # (A.T@W@A)^-1@A.T@W@b

    # x = A.T @ diag_W
    #
    # x_t = x @ A
    # y = np.linalg.inv(x_t)
    # w = A.T @ W
    # z = x @ w @ b

    x = np.linalg.inv(A.T @ diag_W @ A) @ (A.T @ diag_W) @ b
    return x

    # substitution
    # B = ( diag_W ** 0.5 ) @ A
    # c = ( diag_W ** 0.5 ) @ b

    # becomes the ordinary least squares problem:
    # min(B@x-c).T@(B@x-c)

    # B.T@B@x = B.T@c

    # QR decomposition
    # Q, R = np.linalg.qr(B)


def generate_coalition_vectors(num_features):
    """
    Generate perturbed data in binary form (coalitions vectors - matrix)

    :param: num_features - number of features of data samples {int}

    :return: coalitions - coalition vectors in matrix (d, num_features) {0,1}
    """

    coalitions = np.array([list(i) for i in itertools.product([0, 1], repeat=num_features)])

    # zero_index = np.array([[1.] * len(coalitions)]).T
    # coalitions = np.concatenate([zero_index, coalitions], axis=1)

    return coalitions


if __name__ == "__main__":
    a = generate_coalition_vectors(3)
    print(a)