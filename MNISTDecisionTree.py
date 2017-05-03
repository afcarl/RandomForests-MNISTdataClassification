import numpy as np
from random import randrange


class TreeNode:
    colno = 0
    value = 0.0
    Left = None
    Right = None
    Labels = {}

class Tree:
    Root = TreeNode()



def getSubsetOflabelsAndValues(datavalues,datalabels,index):
    datavaluessubset = []
    datalabelssubset = []
    for i in index:
        datavaluessubset.append(datavalues[i])
        datalabelssubset.append(datalabels[i])
    return datavaluessubset,datalabelssubset

def getRandomRange(totalsize,subsetsize):
    index = []
    while len(index) < subsetsize :
        i = randrange(totalsize - 1)
        if i not in index:
            index.append(i)
    return index

def CalculateEntropy(valuemap,total):
    entropy = 0.0
    #Normalizing the values
    for key,val in valuemap.items():
        valuemap[key] = valuemap[key]/float(total)
    for key,val in valuemap.items():
       entropy += valuemap[key]*np.log(1.0/valuemap[key])
    return entropy

def CalculateBestGain(datavalues,datalabels,columnno,currententropy):
    best_gain = 0.0
    best_total_r = 0
    best_total_l = 0
    best_value = 0.0

    unique_values = {}

    for row in range(0, len(datavalues)):
        unique_values[datavalues[row][columnno]] = 1

    for keyvalue,value in unique_values.items():
        total_l = 0
        total_r = 0
        gain = 0.0
        map_r = {}
        map_l = {}
        for row in range(0,len(datavalues)):
            if datavalues[row][columnno] <=keyvalue:
                total_l += 1
                if datalabels[row] in map_l:
                    map_l[datalabels[row]] += 1
                else:
                    map_l[datalabels[row]] = 1
            else:
                total_r +=1
                if datalabels[row] in map_r:
                    map_r[datalabels[row]] += 1
                else:
                    map_r[datalabels[row]] = 1


        p1 = float(total_l)/float(len(datalabels))
        p2 = float(total_r)/float(len(datalabels))
        entropy = p1*CalculateEntropy(map_l,total_l) + p2*CalculateEntropy(map_r,total_r)
        gain = currententropy - entropy

        if gain >= best_gain:
            best_gain = gain
            best_value = keyvalue
            best_total_l = total_l
            best_total_r = total_r

    return best_gain ,best_value,best_total_l,best_total_r


def SplitData(datavalues,datalabels,columnno,value):

    leftindexvalues = []
    rightindexvalues = []

    for row in range(0, len(datavalues)):
        if datavalues[row][columnno] <= value:
            leftindexvalues.append(row)
        else:
            rightindexvalues.append(row)

    return  leftindexvalues,rightindexvalues


def GetLeafNode(datalabels):

    counter ={}
    for lbl in datalabels:
        if lbl in counter:
            counter[lbl] +=1
        else:
            counter[lbl] =1
    node = TreeNode()
    node.Labels = counter
    print(node.Labels)
    return node




def BuildTreeUtil(datavalues,datalabels,numberofcolsselected):

    print(" Build Tree Util before start")
    total_columns = len(datavalues[0])
    split_count = numberofcolsselected
    columns_selected = getRandomRange(total_columns,split_count)

    best_gain = 0.0
    best_value = 0.0
    best_total_l = 0
    best_total_r = 0
    best_column = 0

    best_part_l_values = []
    best_part_r_values = []
    best_part_l_labels = []
    best_part_r_labels = []


    currentEntropyMap = {}

    for row in datalabels:
        if row in currentEntropyMap:
            currentEntropyMap[row] +=1
        else:
            currentEntropyMap[row] = 1

    currentEntropy = CalculateEntropy(currentEntropyMap,len(datalabels))

    for c in columns_selected:
         gain,value,total_l,total_r = CalculateBestGain(datavalues,datalabels,c,currentEntropy)
         if gain >= best_gain:
             best_gain = gain
             best_value = value
             best_total_l = total_l
             best_total_r = total_r
             best_column = c


    if best_gain >0.0 and best_total_l>0 and best_total_r>0 :

        node = TreeNode()
        node.value = best_value
        node.column = best_column
        best_part_l,best_part_r = SplitData(datavalues,datalabels,node.column,node.value)
        best_part_l_values,best_part_l_labels = getSubsetOflabelsAndValues(datavalues,datalabels,best_part_l)
        best_part_r_values, best_part_r_labels = getSubsetOflabelsAndValues(datavalues, datalabels, best_part_r)
        node.Left = BuildTreeUtil(best_part_l_values,best_part_l_labels,numberofcolsselected)
        node.Right = BuildTreeUtil(best_part_r_values,best_part_r_labels,numberofcolsselected)
        return node
    return GetLeafNode(datalabels)



def predict(node,input):
    if len(node.Labels.keys()) != 0:
      return node.Labels

    value = input[node.column]
    if value <= node.value and node.Left != None:
        return predict(node.Left,input)
    elif node.Right != None:
        return  predict(node.Right,input)

    return {}

def BuildTree(datavalues,datalabels,samplecount,selectedfeaturecount):
    tree = Tree()

    samples = []
    samples_labels= []
    index = []

    while len(index)<samplecount:
         j = randrange(0,len(datavalues))
         if j not in index:
            index.append(j)

    for i in index:
        samples.append(datavalues[i])
        samples_labels.append(datalabels[i])

    print(" Build Tree before start")
    tree.Root = BuildTreeUtil(samples,samples_labels,selectedfeaturecount)
    print(" Build Tree before send" )
    return tree


def predictTree(tree,input):
    return predict(tree.Root,input)






















