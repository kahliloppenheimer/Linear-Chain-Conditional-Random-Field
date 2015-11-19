# Linear Chain Conditional Random Field

This is my implementation of a linear chain conditional random field (CRF) for sequence modeling. The dataset and test cases deal with detecting word boundaries in the Thai language.

# Comparison against maximum entropy classifier
I have also included a maximum entropy (logistic regression) classifier implementation to compare results against the CRF. The CRF managed to achieve 83% accuracy for both the Character and Character2 feature sets (defined in corpus.py), while the maximum entropy classifier managed to achieve 88% and 93% accuracy for the Character and Character2 feature sets respectively.

# How to use classifier (from test_word_segmentation.py)
```python

from corpus import Character2, ThaiWordCorpus
from crf import CRF, sequence_accuracy

# Read data into corpus and intialize classifier
corpus = ThaiWordCorpus('orchid97_features.bio', Character2)
crf = CRF(corpus.label_codebook, corpus.feature_codebook)

# Split data into train/dev/test
train = corpus[0:20000]
dev = corpus[20000:20050]
test = corpus[21000:23000]

# Train classifier, then evaluate accuracy on test set
crf.train(train, dev)
accuracy = sequence_accuracy(crf, test)

```


