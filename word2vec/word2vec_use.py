# example: word2vec on wikipedia sentences

from gensim.models import word2vec

# load the model
model = word2vec.Word2Vec.load('300features_40minwords_10context')

print "type %s shape %s" % (type(model.syn0), model.syn0.shape)

# print the vector representation (a 1x300 numpy array)
# print model['anarchism']

# we've used only a small amount of raining data, but still related terms should come out
print model.most_similar(positive=['woman', 'prince'], negative=['man'])

# useful for sentiment analysis?
print model.most_similar("angry")

# which word is least like the others
print model.doesnt_match("breakfast cereal dinner lunch".split())

# normalized euclidean distance?
print model.similarity('woman', 'man')

