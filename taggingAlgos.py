import numpy as np
import sys
from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like



from numpy.lib.npyio import loadtxt
from numpy.lib.shape_base import column_stack




train_x_path,train_y_path,test_x_path,output_log_name= sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]

def FormatFileDataToArray(file):
    
    data_array=np.loadtxt(file,delimiter=',')
    return data_array
    
def add_bias(vectors):
    vectors_list=vectors.tolist()
    for vectorIndex,vector in enumerate(vectors_list):
        vectors_list[vectorIndex]=np.append(vector,1.0)
    vectorsArray=np.array(vectors_list)
    return vectorsArray

def zscore(x_train_array, x_test_array):
    avg = np.mean(x_train_array, axis=0)
    std = np.std(x_train_array, axis=0)
    
    for vectorIndex,x in enumerate(x_train_array):
        x_train_array[vectorIndex] = np.divide((x - avg), std)
        
    for vectorIndex2,x2 in enumerate(x_test_array):
        x_test_array[vectorIndex2] = np.divide((x2 - avg), std)
    return x_train_array, x_test_array

def MostCommonTagInAList(distanceListWithIndexTopK,train_y_array):
    tagsList=[0,0,0]
    for distanceAndIndex in distanceListWithIndexTopK:
        if train_y_array[distanceAndIndex[1]]==0:
            tagsList[0]=tagsList[0]+1
        elif train_y_array[distanceAndIndex[1]]==1:
            tagsList[1]=tagsList[1]+1
        elif train_y_array[distanceAndIndex[1]]==2:
            tagsList[2]=tagsList[2]+1
        else:
            print("error,couldnt find correct y value in the array")
    return tagsList.index(max(tagsList))


def KNNAlgo(train_tuple,test_tuple):
    k=5
    test_x_array = test_tuple[0]
    test_y_array = test_tuple[1]
    y_test_Answers=[]
    for index,untaggedFeatures in enumerate(test_x_array):
        untaggedFeaturesPrediction=KNNApredictorForUntaggedFeature(train_tuple,untaggedFeatures,k)
        y_test_Answers.append(untaggedFeaturesPrediction)
    return y_test_Answers

def KNNApredictorForUntaggedFeature(train_tuple,untaggedFeatures,k):
    train_x_array= train_tuple[0]
    train_y_array= train_tuple[1]
    
    distanceListWithIndex = []
    """ append the tuple <distance between the two features list,index of the tagged feature> into a list """
    for index,featuresList in enumerate(train_x_array):
        distanceListWithIndex.append((np.linalg.norm(featuresList-untaggedFeatures),index))
    """ sort the features in the closest to the untagged feature oreder """
    distanceListWithIndex.sort(key= lambda distance:distance[0])
    """ slice and get the first k """
    distanceListWithIndexTopK=distanceListWithIndex[:k]
    """ count what tag is the most common"""
    mostCommonTag=MostCommonTagInAList(distanceListWithIndexTopK,train_y_array)

    return mostCommonTag

def PerceptronAlgo(train_tuple,test_tuple):
    epoch= 10000
    learning_rate = 1/1000
    test_x_array = test_tuple[0]
    test_y_array = test_tuple[1]

    weightsFromTraining=PerceptronTrainer(train_tuple,epoch,learning_rate)
    y_test_Answers=[]
    for features in (test_x_array):
        y_test_Answers.append(PreceptronTagger(weightsFromTraining,features))
    
    return y_test_Answers

def PerceptronTrainer(train_tuple,epoch,learning_rate):
    weightsVector= np.zeros((3,6))

    for iteration in range(epoch):
        indexes=np.arange(train_tuple[0].shape[0])
        np.random.shuffle(indexes)
        for x_vector,y_correct_tag in zip(train_tuple[0][indexes],train_tuple[1][indexes]):
            y_tagged = PreceptronTagger(weightsVector,x_vector)
            if y_tagged!= int(y_correct_tag):
                weightsVector[int(y_correct_tag)] = weightsVector[int(y_correct_tag)] + learning_rate*x_vector 
                weightsVector[y_tagged] = weightsVector[y_tagged] - learning_rate*x_vector
    return weightsVector

def PreceptronTagger(weightsVector,x_vector):
    tag0Dot = np.dot(weightsVector[0],x_vector)
    tag1Dot = np.dot(weightsVector[1],x_vector)
    tag2Dot = np.dot(weightsVector[2],x_vector)

    tagValue=np.argmax(np.array([tag0Dot,tag1Dot,tag2Dot]))
    return tagValue


def SVMAlgo(train_tuple,test_tuple):
    epoch = 10000
    learning_rate= 1/10000
    error_contant = 3
    test_x_array = test_tuple[0]
    test_y_array = test_tuple[1]
    weightsFromTraining = SVMTrainer(train_tuple,epoch,learning_rate,error_contant)
    y_test_Answers=[]
    for features in (test_x_array):
        y_test_Answers.append( SVMTagger(weightsFromTraining,features))
    
    return  y_test_Answers

def SVMTrainer(train_tuple,epoch,learning_rate,error_contant):
    weightsVector= np.zeros((3,6))
    for iteration in range(epoch):
        indexes=np.arange(train_tuple[0].shape[0])
        np.random.shuffle(indexes)
        for x_vector,y_correct_tag in zip(train_tuple[0][indexes],train_tuple[1][indexes]):
            y_tagged=SVMTagger(weightsVector,x_vector)
            loseValue= SVMLoseCalculator(y_correct_tag,weightsVector,x_vector)
            if loseValue>0:
                weightsVector[int(y_correct_tag)] = weightsVector[int(y_correct_tag)]*(1-learning_rate*error_contant) + learning_rate * x_vector
                weightsVector[int(y_tagged)] = weightsVector[int(y_tagged)]*(1-learning_rate*error_contant) - learning_rate * x_vector
                for classindex,weightVector in enumerate(weightsVector):
                    if classindex!= int(y_correct_tag) and classindex != int(y_tagged):
                        weightsVector[classindex] = weightsVector[classindex]*(1-learning_rate*error_contant)
            else:
                for classindex,weightVector in enumerate(weightsVector):
                    if classindex!= int(y_correct_tag) and classindex != int(y_tagged):
                        weightsVector[classindex] = weightsVector[classindex]*(1-learning_rate*error_contant)
    return weightsVector

def SVMLoseCalculator(y_correct_tag,weightsVector,x_vector):
    loseValues=[]
    for classIndex,classWeight in enumerate(weightsVector):
        if classIndex!=y_correct_tag:
            loseValues.append(max(0,1-np.dot(weightsVector[int(y_correct_tag)],x_vector)+np.dot(classWeight,x_vector)))
    maxLoseValue=max(loseValues)
    return maxLoseValue

def SVMTagger(weightsVector,x_vector):
    tag0Dot = np.dot(weightsVector[0],x_vector)
    tag1Dot = np.dot(weightsVector[1],x_vector)
    tag2Dot = np.dot(weightsVector[2],x_vector)

    tagValue=np.argmax(np.array([tag0Dot,tag1Dot,tag2Dot]))
    return tagValue


def PaAlgo(train_tuple,test_tuple):

    test_x_array = test_tuple[0]
    weightsFromTraining = PaTrainer(train_tuple)
    y_test_Answers=[]
    for features in (test_x_array):
        y_test_Answers.append(PaTagger(weightsFromTraining,features))
    return  y_test_Answers


def PaTrainer(train_tuple):
    weightsVector= np.zeros((3,6))
    
    indexes=np.arange(train_tuple[0].shape[0])
    np.random.shuffle(indexes)
    for x_vector,y_correct_tag in zip(train_tuple[0][indexes],train_tuple[1][indexes]):
        y_tagged=PaTagger(weightsVector,x_vector)
        loseValue= PaLoseCalculator(y_correct_tag,weightsVector,x_vector)
        delta = (loseValue)/(2*np.linalg.norm(x_vector))
        if loseValue>0:
            weightsVector[int(y_correct_tag)]=  weightsVector[int(y_correct_tag)] + delta*x_vector
            weightsVector[y_tagged]=  weightsVector[y_tagged] - delta*x_vector
    return weightsVector
        
def PaLoseCalculator(y_correct_tag,weightsVector,x_vector):
    loseValues=[]
    for classIndex,classWeight in enumerate(weightsVector):
        if classIndex!=y_correct_tag:
            loseValues.append(max(0,1-np.dot(weightsVector[int(y_correct_tag)],x_vector)+np.dot(classWeight,x_vector)))
    loseValues=max(loseValues)
    return loseValues

def PaTagger(weightsVector,x_vector):
    tag0Dot = np.dot(weightsVector[0],x_vector)
    tag1Dot = np.dot(weightsVector[1],x_vector)
    tag2Dot = np.dot(weightsVector[2],x_vector)

    tagValue=np.argmax(np.array([tag0Dot,tag1Dot,tag2Dot]))
    return tagValue



def TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path):
    x_train_array= FormatFileDataToArray(train_x_path)
    y_train_array= FormatFileDataToArray(train_y_path)

    x_test_array=FormatFileDataToArray(test_x_path)
    y_test_array= np.full((np.shape((x_test_array))[0]),3)

    x_train_array,x_test_array=zscore(x_train_array,x_test_array)

    train_tuple=(add_bias(x_train_array),y_train_array)
    test_tuple= (add_bias(x_test_array),y_test_array)
    
    return train_tuple,test_tuple
    

"""for testing only. requires importing "from sklearn.model_selection import train_test_split"""
def TrainAndTestArrayMakerFromTrainOnly(train_x_path,train_y_path):
    x_train_array= FormatFileDataToArray(train_x_path)
    y_train_array= FormatFileDataToArray(train_y_path)

    x_train_array,x_test_array,y_train_array,y_test_array =  train_test_split(x_train_array, y_train_array, test_size = 0.2)

    x_train_array,x_test_array=zscore(x_train_array,x_test_array)

    train_tuple=(add_bias(x_train_array),y_train_array)
    test_tuple= (add_bias(x_test_array),y_test_array)
    return train_tuple,test_tuple

def WriteAnswersToFile(output_log_name,KNNAnswer,PreceptronAnswer,SVMAnswer,PaAnswer):
    with open(output_log_name,'w') as file:
        for indexAnswer,answer in enumerate(KNNAnswer):
            file.write(f"knn: {KNNAnswer[indexAnswer]}, perceptron: {PreceptronAnswer[indexAnswer]}, svm: {SVMAnswer[indexAnswer]}, pa: {PaAnswer[indexAnswer]}\n")

    return

def CalculateSuccsessRate(algoTags,rightTags):
    right_answers=0
    wrong_answers=0
    for index,answer in enumerate(algoTags):
        if answer==rightTags[index]:
            right_answers+=1
        else:
            wrong_answers+=1
    return ((right_answers/(wrong_answers+right_answers))*100)

"""
KNNAnswersPercentage=[]
for times in range(30):
    train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
    KNNAnswer= KNNAlgo(train_tuple,test_tuple)
    KNNAnswersPercentage.append((KNNAnswer,CalculateSuccsessRate(KNNAnswer,test_tuple[1])))
bestKNNAnswer=max(KNNAnswersPercentage,key= lambda answer_tuple:answer_tuple[1])

percAnswersPercentage=[]
for times in range(5):
    train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
    percAnswer= PerceptronAlgo(train_tuple,test_tuple)
    percAnswersPercentage.append((percAnswer,CalculateSuccsessRate(percAnswer,test_tuple[1])))
bestPercAnswer=max(percAnswersPercentage,key= lambda answer_tuple:answer_tuple[1])

SVMAnswersPercentage=[]
for times in range(5):
    train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
    SVMAnswer= SVMAlgo(train_tuple,test_tuple)
    SVMAnswersPercentage.append((SVMAnswer,CalculateSuccsessRate(SVMAnswer,test_tuple[1])))
bestSVMAnswer=max(SVMAnswersPercentage,key= lambda answer_tuple:answer_tuple[1])

PaAnswersPercentage=[]
for times in range(30):
    train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
    PaAnswer= KNNAlgo(train_tuple,test_tuple)
    PaAnswersPercentage.append((PaAnswer,CalculateSuccsessRate(PaAnswer,test_tuple[1])))
bestPaAnswer=max(PaAnswersPercentage,key= lambda answer_tuple:answer_tuple[1])
"""
"""TEST
print(bestKNNAnswer[1])
print(bestPercAnswer[1])
print(bestSVMAnswer[1])
print(bestPaAnswer[1])
WriteAnswersToFile(output_log_name,bestKNNAnswer[0],bestPercAnswer[0],bestSVMAnswer[0],bestPaAnswer[0])
"""


train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
KNNAnswer= KNNAlgo(train_tuple,test_tuple)

train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
percAnswer= PerceptronAlgo(train_tuple,test_tuple)

train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
SVMAnswer= SVMAlgo(train_tuple,test_tuple)

train_tuple,test_tuple=TrainAndTestArrayMaker(train_x_path,train_y_path,test_x_path)
PaAnswer= KNNAlgo(train_tuple,test_tuple)

WriteAnswersToFile(output_log_name,KNNAnswer,percAnswer,SVMAnswer,PaAnswer)
