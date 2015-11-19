# -*- mode: Python; coding: utf-8 -*-
from __future__ import division
from collections import defaultdict
import numpy as np
import math

class MaxEnt(object):

    def __init__(self, label_codebook, feature_codebook):
        self.NUM_ITERATIONS = 5 # Fixed number of iterations of SGD
        self.label_codebook = label_codebook
        self.feature_codebook = feature_codebook
        self.num_features = len(feature_codebook)
        self.labelsToWeights = None

    def get_model(self): return self.labelsToWeights;
    def set_model(self, model): self.labelsToWeights = model
    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        instances = [element for sequence in instances for element in sequence]
        dev_instances = [element for sequence in dev_instances for element in sequence]

        self.train_sgd(instances, dev_instances, 0.001, 30)

    # Trains this classifier using stochastic gradient descent
    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        self.labelsToWeights = self.initializeWeights(train_instances)
        for j in range(self.NUM_ITERATIONS):
            for i in range(0, len(train_instances), batch_size):
                batch = train_instances[i : (i + batch_size)]
                gradient = self.gradient(batch)
                for label in self.labelsToWeights:
                    self.labelsToWeights[label] += learning_rate * gradient[label]
            print 'negLogLikelihood = ',self.negLogLikelihood(dev_instances)

    # Classifies the given instance as the most likely label from the dataset,
    # given the current model
    def classify(self, element):
        posteriors = {}
        for label in self.labelsToWeights:
            posteriors[label] = self.posterior(label, element.feature_vector)
        return max(posteriors, key=posteriors.get)

    # Initializes model parameter weights to zero
    def initializeWeights(self, train_instances):
        labels = {}
        for instance in train_instances:
            if instance.label_index not in labels:
                labels[instance.label_index] = np.zeros(self.num_features)
        return labels

    # Returns the posterior probability P(label | featureVec)
    def posterior(self, label, featureVec):
        dotProds = {}
        # Calculate each posterior once
        for l, w in self.labelsToWeights.iteritems():
            dotProds[l] = math.exp(sum(w[featureVec]))
        return dotProds[label] / sum(dotProds.itervalues())

    # Returns the observed counts for each feature in the passed mini-batch
    def observedCounts(self, instances):
        observed_counts = defaultdict(lambda: np.zeros(self.num_features))
        for instance in instances:
            observed_counts[instance.label_index][instance.feature_vector] += 1
        return observed_counts

    # Returns the expected model counts (right hand sand of gradient difference)
    # given a mini batch of instances
    def expectedModelCounts(self, instances):
        expectedCounts = defaultdict(lambda: np.zeros(self.num_features))
        for instance in instances:
            for label, w in self.labelsToWeights.iteritems():
                posterior = self.posterior(label, instance.feature_vector)
                expectedCounts[label][instance.feature_vector] += posterior
        return expectedCounts

    # Computes the gradient over the given instances
    def gradient(self, instances):
        expected = self.expectedModelCounts(instances)
        observed = self.observedCounts(instances)
        gradient = defaultdict(lambda: np.zeros(self.num_features))
        for label in self.labelsToWeights:
            gradient[label] = observed[label] - expected[label]
        return gradient

    # Computes the negative log-likelihood over a set of instances
    def negLogLikelihood(self, instances):
        return -1 * sum([math.log(self.posterior(instance.label_index, instance.feature_vector)) for instance in instances])

    # Computes the accuracy of the classifier over a set of instances
    def accuracy(self, instances):
        return sum([instance.label_index == self.classify(instance) for instance in instances]) / len(instances)

    # Computes the accuracy of this classifier over the sequences in the test set
    def sequence_accuracy(self, test_set):
        correct = 0.0
        total = 0.0
        for sequence in test_set:
            decoded = [self.classify(char) for char in sequence]
            assert(len(decoded) == len(sequence))
            total += len(decoded)
            for i, instance in enumerate(sequence):
                if instance.label_index == decoded[i]:
                    correct += 1
        return correct / total
