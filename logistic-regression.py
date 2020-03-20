from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#sample = all except last two column(label<=50k, label>50k)
#target = the last column(label>50k)
X1 = one_hot_df.iloc[:,:-2]
y1 = df.iloc[:,-1]
X2 = one_hot_var1.iloc[:,:-2]
y2 = var1.iloc[:,-1]
X3 = one_hot_var2.iloc[:,:-2]
y3 = var2.iloc[:,-1]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.5)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.5)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,test_size=0.5)

classifier_df = LogisticRegression()
classifier_var1 = LogisticRegression()
classifier_var2 = LogisticRegression()

#classifier_df.fit(sample, target)
classifier_df.fit(X1_train, y1_train.ravel())
classifier_var1.fit(X2_train, y2_train.ravel())
classifier_var2.fit(X3_train, y3_train.ravel())

print("initial data set model score:", classifier_df.score(X1_test, y1_test))
print("data set(1) model score:", classifier_var1.score(X2_test, y2_test))
print("data set(2) model score:", classifier_var2.score(X3_test, y3_test))