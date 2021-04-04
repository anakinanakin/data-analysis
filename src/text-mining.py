# nltk's default stoplist:
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))

import csv
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

train_data = []

with open('./Assignment2Datasets/pg_train.csv', encoding = 'latin-1') as csvfile:
    #all lowercase
    lower_stream = (line.lower() for line in csvfile)
    pg_train = csv.reader(lower_stream, delimiter=' ', quotechar='"', escapechar='^')
    
    #no punctuation
    for row in pg_train:
        row = re.sub('[^A-Za-z0-9#\w\s]+', '', str(row))
        
        #tokenize
        row = nltk.word_tokenize(row)
        
        #stemming
        port = nltk.PorterStemmer()
        for word in row:
            word = port.stem(word)
            
        #stopword removal
        row = [word for word in row if word not in stoplist]
        
        #make wordlist into string
        row = ' '.join(row)
        #make a list with string elements
        train_data.append(row)

#print train_data
#make train_data to list for input
#train_data = [train_data]

authorList_train = []
train_data_new = []
blank = 0

#separate author and content
for s in train_data:
    #avoid newline 's'
    if blank == 1:
        blank = 0
        continue
    blank += 1
    
    author = ""
    ctr = 0
    for char in s:
        if char != "#":
            author += char
            ctr += 1
            #not working
            #s = s.replace(char, "", 1)
            #print char
            #print ctr
            #s = ''.join(s.split(char, 1))
        else:
            #s = s.replace(char, "", 1)
            #s = ''.join(s.split(char, 1))
            break
    ctr += 2
    train_data_new.append(s[ctr:])
    #s = s.lstrip(author)
    #cannot access s
    #s += "00000000000"
    #print (s)
    authorList_train.append(author)
    
#print (authorList_train)
#print (train_data_new)

#binary document-term matrix
count_vect_binary = CountVectorizer(binary=True)
X = count_vect_binary.fit_transform(train_data_new)

#print count_vect.get_feature_names()
#print X.toarray()

#train logistic clf
log_binary = Pipeline([('vect', count_vect_binary), ('logistic', LogisticRegression())])
log_binary = log_binary.fit(train_data_new, authorList_train)