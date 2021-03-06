# example: word2vec on wikipedia sentences
# note that tsne may not give realistic output when the dataset is small or the vector space is relatively large

from collections import defaultdict

from gensim.models import word2vec
import numpy as np
import matplotlib.pyplot as plt
import tsne as tsne

class tsnemodel:
    'tsnemodel'
    vecs = defaultdict(list)
    colors = defaultdict(list)
    labels = defaultdict(list)

    def __init__(self, model):
       self.model = model

    def add(self, groupname, color, words):
        vecs = []
        for word in words:
            try:
               self.vecs[groupname].append(model[word])
               self.labels[groupname].append(word)
               self.colors[groupname] = color
            except KeyError:
               continue

    def tsne(self):
        vecs = []
        labels = []

        for key, value in self.vecs.iteritems():
            vecs += value
            labels += self.labels[key]

        vecs = np.array(vecs, dtype='float64') #TSNE expects float type values
        # call tsne with (vectors, #output dimensions (2=2D), intermediate dimensions (not sure what this does), perplexity)
        # perplexity modifies the repulsion between vectors, so a high value 
        # distributes nodes evenly over space, while a low value groups values
        self.t = tsne.tsne(vecs, 2, 2, 4)
        vec_group_start = 0;
        for key, value in self.vecs.iteritems():
            color = self.colors[key]
            for j in range(len(value)):
                index = vec_group_start + j
                label = self.labels[ key ][j]
                plt.plot(self.t[ index ][0], self.t[ index ][1])
                plt.text(self.t[ index ][0], self.t[ index ][1], label, color=color, horizontalalignment='center')
                #print self.t[ index ][0], self.t[ index ][1]
            vec_group_start += len(value)
        plt.show()
        return plt

# load the model
model = word2vec.Word2Vec.load('300features_40minwords_10context')

t = tsnemodel(model)
t.add('royalty', 'b', ['king', 'queen', 'prince', 'princess', 'monarch', 'castle'])
t.add('food', 'g', ['bread', 'wine', 'meat', 'butter', 'steak', 'pork', 'beer'])
t.add('animals', 'r', ['monkey', 'wolf', 'lion'])

t.tsne()

