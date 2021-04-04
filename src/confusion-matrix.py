import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# model 7. (c, i)
from sklearn.neural_network import MLPClassifier

features = df.columns[:-1]
X = df[features]
X = pd.get_dummies(X)
Y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.5)
nn = MLPClassifier()
a = nn.fit(X_train, y_train)

pred_y = nn.predict(X_test)

import numpy as np
pred_y = np.array(pred_y)
true_y = np.array(y_test)

cm = confusion_matrix(true_y, pred_y)
tn, fp, fn, tp = cm.ravel()
print cm.ravel()

#question 6 (c)
#sample = all except last two column(label<=50k, label>50k)
#target = the last column(label>50k)
X1 = one_hot_df.iloc[:,:-2]
y1 = df['label']

X_train, X_test, y_train, y_test = train_test_split(X1,y1,test_size=0.5)
classifier_df = LogisticRegression()
#classifier_df.fit(sample, target)
classifier_df.fit(X_train, y_train.ravel())

pred_y = classifier_df.predict(X_test)

pred_y = np.array(pred_y)
true_y = np.array(y_test)

cm = confusion_matrix(true_y, pred_y)
tn, fp, fn, tp = cm.ravel()
print cm.ravel()
