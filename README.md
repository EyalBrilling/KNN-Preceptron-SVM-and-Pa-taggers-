# KNN,Preceptron,SVM and Pa taggers
implementation of the following algorithms from zero using numpy.

# The data

The dataset to be learnt and tagged is Iris flowers classification. There are 3 Iris flower species with five features per instance. 
In the train_x file there are 240 Iris flower feature instance to learn from. Each instance is a feature vector which contains 5 features seperated by commas. each flower instance is seperated by newline.
the corresponding train_y file contains the correct labels(species type). The tags start at 0( tags are 0,1 or 2). Each tag is seperated by newline.

# The output

For each flower instance to be tagged by the different algorithms,the code will output each different algorithm predection. The following format is used:
```
knn: predection, perceptron: predection, svm: predection, pa: predection \n
```
Where predection is the predected tag by the corrspending algorithm,from the range 0 to 2.

****Notice: order of the test input is saved. If an instane of a flower is on line X on the test file,it will be on line X in the predection file output****

# Program arguments
the program expectes 4 arguments from the user
1) The ****training set feature vectors file path****  (train_x.txt can be used) 
2) The ****training set tags file path****. (train_y.txt can be used) 
3) The ****test set feature vectors file path**** ( test_x.txt can be used)
4) The ****algorithms results file path****
