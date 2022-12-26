from explanation import Explanation
from naive_bayes_classif.filter import *
from naive_bayes_classif.quality import compute_quality_for_corpus
import pprint

# train a (naive-bayes) classifier
classif = Filter()
classif.train("data/1")

# showing (naive-bayes) classifier quality
classif.test("data/2")
result = compute_quality_for_corpus("data/2")

# visualize (naive-bayes) classifier interpretability
path_to_instance = "data/2/01286.80a17353e03cab185cb52237b60359e4.txt"
probs, relevant_words = classif.test(path_to_instance)

pprint.pprint(relevant_words)

exp_interpretation_bayes = Explanation(relevant_words, plot_title="Naive Bayes Classifier Interpretation", x_label="partial_probabilities")
exp_interpretation_bayes.graph_plot()


# 01286.80a17353e03cab185cb52237b60359e4


# explain the predictions using LIME
# TODO
# explainer = LimeTextExplainer(class_names)
# explanation = explainer.explain_instance(data[idx],...)

# visualize the explanations
# TODO
