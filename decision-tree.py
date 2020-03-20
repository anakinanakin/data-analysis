# your code
from sklearn import tree
from subprocess import check_output, check_call

#descriptive features
#only use categorical attributes with one hot encoding
X = df[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']] 
X = pd.get_dummies(X)
#target feature
Y = df[["label"]]

entropy_tree = tree.DecisionTreeClassifier(criterion="entropy")   
entropy_tree.fit(X, Y)
gini_tree = tree.DecisionTreeClassifier(criterion="gini")   
gini_tree.fit(X, Y)

#visualize in pdf
column_names = list(X.columns.values)
dot_file_entropy = "entropy.dot"
pdf_file_entropy = "entropy.pdf"
dot_file_gini = "gini.dot"
pdf_file_gini = "gini.pdf"

with open(dot_file_entropy, "w") as f:
    f = tree.export_graphviz(entropy_tree, out_file=f, 
                                 feature_names= column_names, 
                                 class_names=["label<=50K", "label>50K"], 
                                 filled=True, rounded=True)

    
with open(dot_file_gini, "w") as f:
    f = tree.export_graphviz(gini_tree, out_file=f, 
                                 feature_names= column_names, 
                                 class_names=["label<=50K", "label>50K"], 
                                 filled=True, rounded=True)

#convert dot to pdf file
try:
    #check_output("dot -Tpdf "+ dot_file + " -o " + pdf_file , shell=True)
    check_call(['dot','-Tpdf','entropy.dot','-o','entropy.pdf'])
    check_call(['dot','-Tpdf','gini.dot','-o','gini.pdf'])
except:
    print("Make sure that you have installed Graphviz, otherwise you can not see the visual tree. But you can find descriptions in a dot file")