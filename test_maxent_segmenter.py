from corpus import Character, Character2, CharacterTest, ThaiWordCorpus
from maxent import MaxEnt
from unittest import TestCase, main

class TestSegmenting(TestCase):

    def setUp(self):
        self.corpus = ThaiWordCorpus('orchid97_features.bio', Character)
        self.tagger = MaxEnt(self.corpus.label_codebook, self.corpus.feature_codebook)

    def test_segmenting(self):
        train = self.corpus[0:20000]
        dev = self.corpus[20000:20050]
        test = self.corpus[21000:23000]
        self.tagger.train(train, dev)

        accuracy = self.tagger.sequence_accuracy(test)
        print 'final acc = ', accuracy
        self.assertGreaterEqual(accuracy, 0.80)

if __name__ == '__main__':
    main(verbosity=2)


