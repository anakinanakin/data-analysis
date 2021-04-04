from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules as arule

fpb_data_1 = cluster_dataset_new.copy()
fpb_data_2 = cluster_dataset_new.copy()
ctr = 0
        
for index, row in fpb_data_1.iterrows(): 
    #cluster label = 0
    if cluster_labels[ctr] == 1:
        fpb_data_1.drop(index, inplace=True)
    ctr+=1
    
ctr = 0
for index, row in fpb_data_2.iterrows():
    #cluster label = 1
    if cluster_labels[ctr] == 0:
        fpb_data_2.drop(index, inplace=True)
    ctr+=1

#show the data set prepared for each cluster
print ("fpb_data_1:")
print (fpb_data_1)
print ("fpb_data_2:")
print (fpb_data_2)

frequent_itemsets1 = apriori(fpb_data_1, min_support = 0.2, use_colnames = True)
print ("frequent_itemsets1:")
print (frequent_itemsets1)
frequent_itemsets2 = apriori(fpb_data_2, min_support = 0.2, use_colnames = True)
print ("frequent_itemsets2:")
print (frequent_itemsets2)