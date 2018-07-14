#binary classification with naive bayes
import numpy as np
from collections import Counter



def count_vectorize(features):
    """
    Input: feature array 
    Output: vocabulary dictionary, inverse vocabulary dictionary, feature vectors
    Takes a collection of documents and vectorizes the text
    """
    #only counts no n-grams
    vocab = {}
    vocab_rev = {}
    #extract the vocabulary
    for text in features:
        #tokenize text (now just split)
        tok = text.split()
        for word in tok:
            if word not in vocab:
                vocab_size = len(vocab)
                vocab[word] = vocab_size
                vocab_rev[vocab_size] = word

    #do the vectorization
    vect_features = []
    for text in features:
        word_counts = Counter(text.split())
        vect_features.append([word_counts[vocab_rev[i]] for i in range(len(vocab))])
    
    return vocab, vocab_rev, vect_features


def test_vectorize(vocab, test):
    """
    Input: vocabulary dictionary, test dataset
    Output: test dataset vectorized
    Vectorizes the test dataset according to the vocabulary dictionary
    """
    test_vect = []
    for doc in test[0]:
        #print(doc)
        tok = doc.split()
        doc_vect = [0 for i in range(len(vocab))]
        for word in tok:
            try:
                doc_vect[vocab[word]] += 1
            except:
                print(word, 'is not in the vocabulary dictionary. Ignore this word, and move on.')
                continue
        test_vect.append(doc_vect)
    return test_vect
    
def frequency_table(target, features):
    """
    Input: target and vectorized features
    Output: frequency table 
    Creates a table of the co-ocurrence between each feature vector dimension and each
    outcome (plus one to account for terms that don't appear with certain outcomes)
    """
    n_target_classes = len(set(target))
    n_feature_classes = len(features[0])
    print(n_feature_classes, n_target_classes)
    frequency_table = np.ones((n_feature_classes, n_target_classes))
    for cx, x in enumerate(features):
        #print(x)
        for cx_i, x_i in enumerate(x):
            frequency_table[cx_i, target[cx]] += 1*x_i
    return frequency_table


def conditional_probs(frequency_table, vocab_rev):
    """
    Input: Frequency table, inverse vocabulary dict (index -> word)
    Output: Conditional probability table
    Creates the conditional probability table (the probability of observing an outcome given 
    an observation of the feature vector element)
    """
    class_lengths = np.sum(frequency_table, axis = 0)
    vocab_size = len(vocab_rev)
    print(class_lengths)
    cond_probs = frequency_table/(class_lengths)# + vocab_size)
    print(cond_probs)
    #total_events = np.sum(frequency_table)
    #likelihood_target = np.sum(frequency_table, axis = 0)/total_events
    #likelihood_class = np.sum(frequency_table, axis = 1)/total_events
    return cond_probs

def target_probs(target):
    """
    Input: target array
    Output: target class probability
    Returns the probability of observing each class from the target array
    """
    counts = Counter(target)
    class_prob = [0 for _ in range(len(counts))]
    for t in counts:
        class_prob[t] = counts[t]/len(target)
    return class_prob


def classify(data, cond_prob, class_prob):
    """
    Input: Vectorized test feature array, conditional probability table, class probability array
    Output: predicted class array
    Takes the feature vectors for new data and predicts the class
    """
    pred = []
    for doc in data:
        prior = np.log(class_prob)
        for outcome in range(len(prior)):
            for cx_i, x_i in enumerate(doc):
                prior[outcome] += x_i*np.log(cond_prob[cx_i, outcome])
        pred.append(np.argmax(prior))
    return pred







features = ['red red red', 'blue red green', 'green blue red', 'blue blue', 'green', 'orange yellow',
             'yellow yellow', 'purple orange yellow', 'purple purple orange']

target = [0,0,0,0,0,1,1,1,1]

#features = [['chinese beijing chinese', 'chinese chinese shanghai', 'chinese macao', 'tokyo japan chinese']]
#target = [1,1,1,0]

test = ['red', 'red red red', 'red purple red', 'orange purple yellow']

vocab, vocab_rev, vect_features =  count_vectorize(features)
print(vect_features)
print(vocab_rev)
freq_table = frequency_table(target, vect_features)
cond_probs = conditional_probs(freq_table, vocab_rev)
class_prob = target_probs(target)


test_features = test_vectorize(vocab, test)
print(test)
print(classify(test_features, cond_probs, class_prob))
