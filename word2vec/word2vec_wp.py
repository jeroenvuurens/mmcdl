# example: word2vec on wikipedia sentences

from gensim.models import word2vec

# Import various modules for string cleaning
import re
import logging
import gzip
from nltk.corpus import stopwords

def sentence_to_wordlist( sentence, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 2. Remove non-letters
    sentence = re.sub("[^a-zA-Z]"," ", sentence)
    #
    # 3. Convert words to lower case and split them
    words = sentence.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( raw_sentences, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( sentence_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

# Read data from files
# wp are already cleaned from markup
# every sentence is on a different line, an empty line separates different WP pages
# and the first line contains the name of the page
with gzip.open('data/wp_0.txt.gz') as f:
    train = f.readlines()

# remove eol and empty lines
train = [x.strip('\n') for x in train if len(x) > 1]

# Verify the number of reviews that were read (100,000 in total)
print "Read %d sentences\n" % (len(train))

sentences = review_to_sentences(train)

# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set values for various parameters
num_features = 50    # Word vector dimensionality
min_word_count = 5   # Minimum word count
num_workers = 2       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
