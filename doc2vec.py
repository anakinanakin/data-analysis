import gensim
import multiprocessing
from gensim.models import Doc2Vec
from tqdm import tqdm
from sklearn import utils

train_tagged = []

for i in range (0, len(train_data_new)):
    train_tagged.append(gensim.models.doc2vec.TaggedDocument(words=gensim.utils.simple_preprocess(train_data_new[i]), tags=authorList_train[i]))

#print(repr(train_tagged[0]))

cores = multiprocessing.cpu_count()

doc2vec_model = Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
doc2vec_model.build_vocab([x for x in tqdm(train_tagged)])

for epoch in range(30):
    doc2vec_model.train(utils.shuffle([x for x in tqdm(train_tagged)]), total_examples=len(train_tagged), epochs=1)
    doc2vec_model.alpha -= 0.002
    doc2vec_model.min_alpha = doc2vec_model.alpha

from sklearn.linear_model import LogisticRegression

# Building the feature vector for the classifier
def vec_for_learning(model, tagged_docs):
    targets, doc2vec_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])
    return targets, doc2vec_vectors

# Translating docs into vectors for training and test set
y_train, X_train = vec_for_learning(doc2vec_model, train_tagged)

log_reg_doc2vec = LogisticRegression()
log_reg_doc2vec.fit(X_train, y_train)