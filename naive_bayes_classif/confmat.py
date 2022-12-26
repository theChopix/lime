class BinaryConfusionMatrix:

    def __init__(self, pos_tag, neg_tag):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag
        # true positive
        self.tp = 0
        # true negative
        self.tn = 0
        # false positive
        self.fp = 0
        # false negative
        self.fn = 0

    def as_dict(self):
        return {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn}

    def update(self, truth, prediction):
        self.check_input(prediction)
        self.check_input(truth)

        if truth == self.pos_tag and prediction == self.pos_tag:
            self.tp += 1
        elif truth == self.neg_tag and prediction == self.neg_tag:
            self.tn += 1
        elif truth == self.neg_tag and prediction == self.pos_tag:
            self.fp += 1
        elif truth == self.pos_tag and prediction == self.neg_tag:
            self.fn += 1

    def check_input(self, value):
        if value not in (self.pos_tag, self.neg_tag):
            raise ValueError

    def compute_from_dicts(self, truth_dict, prediction_dict):
        for key in truth_dict:
            if key not in prediction_dict:
                raise ValueError
                # continue
            truth = truth_dict[key]
            prediction = prediction_dict[key]
            self.update(truth, prediction)