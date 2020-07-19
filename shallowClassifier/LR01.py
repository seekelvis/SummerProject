
from numpy import *
import matplotlib.pyplot as plt
import csv
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np

#从文件中加载数据：特征X，标签label
def loadDataSet(fileName):
    dataMatrix=[]
    dataLabel=[]


    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        m = 23 #特征数量
        print(m)
        average_data = zeros(m) #记录每个特征的平均值，先累加后除以数量
        valid_num = zeros(m)
        for line in reader:
            # print(line)
            oneline=[]
            for i in range(len(line)-1):
                if (line[i]!=''):
                    oneline.append(float(line[i]))
                    average_data[i] += float(line[i])
                    valid_num[i] += 1
                else:
                    oneline.append(-1)
            dataMatrix.append(oneline)
            dataLabel.append(int(line[23]))

    #calculate average
    for i in range(m):
        average_data[i] = float(average_data[i]) / valid_num[i]
    print("average_data")
    print((average_data))

    #fill the miss data
    for i in range(len(dataMatrix)):
        for j in range(len(dataMatrix[0])):
            if dataMatrix[i][j]==-1:
                dataMatrix[i][j] = average_data[j]

    #print
    for i in range(len(dataMatrix)):
      print(dataMatrix[i])
    print(dataLabel)
    print(mat(dataLabel).transpose())
    matLabel=mat(dataLabel).transpose()

    return dataMatrix,matLabel

#logistic回归使用了sigmoid函数
def sigmoid(inX):
    return 1/(1+exp(-inX))

#函数中涉及如何将list转化成矩阵的操作：mat()
#同时还含有矩阵的转置操作：transpose()
#还有list和array的shape函数
#在处理矩阵乘法时，要注意的便是维数是否对应

#graAscent函数实现了梯度上升法，隐含了复杂的数学推理
#梯度上升算法，每次参数迭代时都需要遍历整个数据集
def graAscent(dataMatrix,matLabel):
    m,n=shape(dataMatrix)
    matMatrix=mat(dataMatrix)

    w=ones((n,1))
    alpha=0.001
    num=500
    for i in range(num):
        error=sigmoid(matMatrix*w)-matLabel
        w=w-alpha*matMatrix.transpose()*error
    return w


#随机梯度上升算法的实现，对于数据量较多的情况下计算量小，但分类效果差
#每次参数迭代时通过一个数据进行运算
#m个样本n个特征
def stocGraAscent(dataMatrix,matLabel):
    m,n=shape(dataMatrix)
    matMatrix=mat(dataMatrix)

    w=ones((n,1))
    alpha=0.001
    num=20  #这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    for i in range(num):
        for j in range(m):
            error=sigmoid(matMatrix[j]*w)-matLabel[j]
            w=w-alpha*matMatrix[j].transpose()*error
    return w

#改进后的随机梯度上升算法
#从两个方面对随机梯度上升算法进行了改进,正确率确实提高了很多
#改进一：对于学习率alpha采用非线性下降的方式使得每次都不一样
#改进二：每次使用一个数据，但是每次随机的选取数据，选过的不在进行选择
def stocGraAscent1(dataMatrix,matLabel):
    m,n=shape(dataMatrix)
    matMatrix=mat(dataMatrix)

    w=ones((n,1))
    num=200  #这里的这个迭代次数对于分类效果影响很大，很小时分类效果很差
    setIndex=set([])
    for i in range(num):
        for j in range(m):
            alpha=4/(1+i+j)+0.01

            dataIndex=random.randint(0,100)
            while dataIndex in setIndex:
                setIndex.add(dataIndex)
                dataIndex=random.randint(0,100)
            error=sigmoid(matMatrix[dataIndex]*w)-matLabel[dataIndex]
            w=w-alpha*matMatrix[dataIndex].transpose()*error
    return w

def LR_predict(weight,testDataMatrix, testMatLabel):
    prediction = []
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(testDataMatrix)):
        tempPre = float(testDataMatrix[i]*weight)
        tempLabel = float(testMatLabel[i])
        prediction.append(tempPre)
        tempPre = round(tempPre)
        print(tempPre,tempLabel)
        if tempPre==1 and tempLabel==1:
            TP +=1
        elif tempPre==0 and tempLabel==0:
            TN +=1
        elif tempPre==1 and tempLabel==0:
            FP +=1
        elif tempPre==0 and tempLabel==1:
            FN +=1
    print("TP:",TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    trueRate = (TP+TN)/(TP+TN+FP+FN)

    print("trueRate:")
    print(trueRate)

    label = np.array(testMatLabel)
    prediction = np.array(prediction)
    fpr, tpr, thresholds = metrics.roc_curve(label,prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("fpr")
    print(fpr)
    print("tpr")
    print(tpr)
    print("thresholds")
    print(thresholds)
    print("auc")
    print(auc)

    return prediction,auc





#绘制图像
def draw(weight):
    x0List=[];y0List=[];
    x1List=[];y1List=[];
    f=open('testSet.txt','r')
    for line in f.readlines():
        lineList=line.strip().split()
        if lineList[2]=='0':
            x0List.append(float(lineList[0]))
            y0List.append(float(lineList[1]))
        else:
            x1List.append(float(lineList[0]))
            y1List.append(float(lineList[1]))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x0List,y0List,s=10,c='red')
    ax.scatter(x1List,y1List,s=10,c='green')

    xList=[];yList=[]
    x=arange(-3,3,0.1)
    for i in arange(len(x)):
        xList.append(x[i])

    y=(-weight[0]-weight[1]*x)/weight[2]
    for j in arange(y.shape[1]):
        yList.append(y[0,j])

    ax.plot(xList,yList)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    # dataMatrix,matLabel=loadDataSet('D:/code/SummerProject/shallowClassifier/data/train01.csv')
    # trainDataMatrix = dataMatrix[0:-1000]
    # trainMatLabel = matLabel[0:-1000]
    # testDataMatrix = dataMatrix[-1000:]
    # testMatLabel =  matLabel[-1000:]
    trainDataMatrix, trainMatLabell = loadDataSet('D:/code/SummerProject/shallowClassifier/data/train01.csv')
    testDataMatrix, testMatLabel = loadDataSet('D:/code/SummerProject/shallowClassifier/data/test01.csv')

    weight = stocGraAscent1(trainDataMatrix, trainMatLabell)
    print("weight")
    print(weight)

    prediction, auc = LR_predict(weight, testDataMatrix, testMatLabel)

    # draw(weight)



    # weight=graAscent(dataMatrix,matLabel)
    # weight=stocGraAscent1(dataMatrix,matLabel)
    # print(weight)
    # draw(weight)