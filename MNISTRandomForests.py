
import pickle
import gzip
import MNISTDecisionTree
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=1)

class Forest:
    trees = []


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')

training_data, validation_data, test_data = pickle.load(f)
f.close()

train_features,train_label = training_data
test_features, test_label = test_data

print(train_features.shape)
print(test_features.shape)




def BuildForest(datavalues,datalabels,treescount,samplecount,selectedcolcount):
    forest = Forest()
    async_result_list = []

    for i in range(0,treescount):
        print("Training  starts on Tree" + str(i))
        async_result = pool.apply_async(MNISTDecisionTree.BuildTree,args= (datavalues,datalabels,samplecount,selectedcolcount))
        async_result_list.append(async_result)

    count = 0
    for async_result in async_result_list:
       count+=1
       forest.trees.append(async_result.get())
       print("Completed Training on Tree"+str(count))

    return forest



def MainForestMethod(datavalues,datalabels,testvalues,testlabels,treescount):
    samplecount = 2000
    selectedcolcount = 30
    forest = BuildForest(datavalues, datalabels, treescount, samplecount, selectedcolcount)
    predictedresults = []
    for i in range(len(testvalues)):
        predictedresults.append(predict(forest,testvalues[i]))

    actualresults = testlabels.tolist()

    Calculateaccuracy(predictedresults,actualresults)

def Calculateaccuracy(predicted,actual):
    count = 0
    actualcount = len(actual)
    for i in range(actualcount):
        if predicted[i] == actual[i]:
            count = count + 1
    accuracy = (count / float(actualcount)) * 100.0

    print("Random Forest Accuracy for MNIST dataset is:" + str(accuracy))



def predict(forest,input):
    countlabels ={}
    for tree in forest.trees:
        labels = MNISTDecisionTree.predictTree(tree,input)
        total = 0.0
        for key,value in labels.items():
            total += float(value)
        for key,value in labels.items():
            temp = float(value) / total
            print(temp)
            if key in countlabels:
                countlabels[key] += temp
            else:
                countlabels[key] = temp

    v = list(countlabels.values())
    k = list(countlabels.keys())

    return k[v.index(max(v))]


#Considering 100 trees in the forest
MainForestMethod(train_features,train_label,test_features,test_label,100)


