## LIME

### Local Interpretable Model-Agnostic Explanations

LIME is concept of explaining single predictions of classification models based on their inputs and outputs (data instance and its prediction).

![Lime2-500x81](https://user-images.githubusercontent.com/90426243/212056261-bb4f0045-f60c-4ef5-b165-c3b98d8472e2.png)


Specific instance of LIME used in this implementation examines spam classification problem and uses Shapley and Banzhaf values as its weights (as game-theoretic concept).

Repository includes:

* Naive Bayesian Classifier - thanks to its simplicity we can derive some inherent interpretability, which is shown in jupyter file

* Explanations on specific instance (in jupyter file) - using shapley weights and banzhaf weights
