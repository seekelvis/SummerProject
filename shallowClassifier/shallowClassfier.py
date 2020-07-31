
from numpy import *
import matplotlib.pyplot as plt
import csv
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    # print("average_data")
    # print((average_data))

    #fill the miss data
    for i in range(len(dataMatrix)):
        for j in range(len(dataMatrix[0])):
            if dataMatrix[i][j]==-1:
                dataMatrix[i][j] = average_data[j]

    #print
    # for i in range(len(dataMatrix)):
    #   print(dataMatrix[i])
    # print(dataLabel)
    # print(mat(dataLabel).transpose())

    # matLabel=mat(dataLabel).transpose()
    dataMatrix = np.array(dataMatrix)
    matLabel = np.array(dataLabel)

    return dataMatrix,matLabel


def SVM(trainDataMatrix,trainMatLabell,testDataMatrix):
    # X = np.array(trainDataMatrix) #训练数据
    # Y = np.array(trainMatLabell) #训练标签
    # T = np.array(testDataMatrix) #预测数据
    X = trainDataMatrix
    Y = trainMatLabell
    T = testDataMatrix
    # print(X.shape,Y.shape)
    svc=SVC(C=0.1,kernel='linear',probability=True) #注意函数参数说明
    svc.fit(X,Y)
    pre=svc.predict_proba(T)
    # print(X.shape,Y.shape)
    # print(pre)
    return pre

def LR(trainDataMatrix,trainMatLabell,testDataMatrix):
    X = trainDataMatrix
    Y = trainMatLabell
    T = testDataMatrix
    # 逻辑回归模型
    log_model = LogisticRegression()
    # 训练逻辑回归模型
    log_model.fit(X, Y)
    # 预测y的值
    # pre = log_model.predict(T)
    pre = log_model.predict_proba(T)
    return pre

def RF(trainDataMatrix,trainMatLabell,testDataMatrix):
    X = trainDataMatrix
    Y = trainMatLabell
    T = testDataMatrix
    # a.tolist()
    # 定义RF模型参数
    rf_model = RandomForestClassifier(n_estimators=200,max_depth=10,max_features=sqrt)
    # 训练RF模型
    rf_model.fit(X, Y)
    # 预测值
    pre = rf_model.predict_proba(T)
    return pre


def analyze(prediction,testMatLabel):
    # TP = 0
    # TN = 0
    # FP = 0
    # FN = 0
    # for i in range(len(prediction)):
    #     print(prediction[i])
    #     tempLabel = int(testMatLabel[i])
    #     tempPre = round(prediction[i])
    #     # print(tempPre, tempLabel)
    #     if tempPre == 1 and tempLabel == 1:
    #         TP += 1
    #     elif tempPre == 0 and tempLabel == 0:
    #         TN += 1
    #     elif tempPre == 1 and tempLabel == 0:
    #         FP += 1
    #     elif tempPre == 0 and tempLabel == 1:
    #         FN += 1
    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)
    # trueRate = (TP + TN) / (TP + TN + FP + FN)
    #
    # print("trueRate:")
    # print(trueRate)

    # pos_prediction=[]
    # for i in range(len(prediction)):
    #     pos_prediction.append(prediction[i,1])
    # pos_prediction=np.array(pos_prediction)

    fpr, tpr, thresholds = metrics.roc_curve(testMatLabel, prediction[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("fpr")
    print(fpr)
    print("tpr")
    print(tpr)
    print("thresholds")
    print(thresholds)
    print("auc")
    print(auc)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.show()
if __name__ == '__main__':

    # trainDataMatrix, trainMatLabell = loadDataSet('D:/code/SummerProject/shallowClassifier/data/train02.csv')
    # testDataMatrix, testMatLabel = loadDataSet('D:/code/SummerProject/shallowClassifier/data/test02.csv')
    #
    # # SVMpre = SVM(trainDataMatrix, trainMatLabell,testDataMatrix)
    # # print("SVMpre")
    # # print(SVMpre)
    # # analyze(SVMpre,testMatLabel)
    #
    #
    # DataMatrix, MatLabell = loadDataSet('D:/code/SummerProject/shallowClassifier/data/data01.csv')
    # trainDataMatrix, testDataMatrix, trainMatLabell,  testMatLabel = model_selection.train_test_split(DataMatrix, MatLabell, test_size=0.2, random_state=27)
    #
    # # sourceDataMatrix, sourceMatLabell = loadDataSet('D:/code/SummerProject/shallowClassifier/data/train01.csv')
    # # targetData, targetLabel = loadDataSet('D:/code/SummerProject/shallowClassifier/data/test01.csv')
    # # aData, testDataMatrix, aLabel, testMatLabel = model_selection.train_test_split(targetData, targetLabel, test_size=0.3, random_state=27)
    # # tempData = sourceDataMatrix.tolist()
    # # tempData.append(aData.tolist())
    # # trainDataMatrix = np.array(tempData)
    # # tempLabel = sourceMatLabell.tolist()
    # # tempLabel.append(aLabel.tolist())
    # # trainMatLabell = np.array(tempLabel)
    # # trainDataMatrix = np.vstack(sourceDataMatrix,aData)
    # # trainMatLabell = np.hstack(sourceMatLabell,aLabel)
    # # print(sourceMatLabell)
    #
    # # #########   LR   ###########
    # # LRpre = LR(trainDataMatrix, trainMatLabell, testDataMatrix)
    # # # print("LRpre")
    # # # print(LRpre)
    # # analyze(LRpre, testMatLabel)
    #
    # ########   SVM   ###########
    # SVMpre = SVM(trainDataMatrix, trainMatLabell,testDataMatrix)
    # # print("SVMpre")
    # # print(SVMpre)
    # analyze(SVMpre,testMatLabel)

    #########   RF  ###########
    # RFpre = RF(trainDataMatrix, trainMatLabell,testDataMatrix)
    # # print("SVMpre")
    # # print(SVMpre)
    # analyze(RFpre,testMatLabel)

    x = [[],[]]
    y = [[],[]]
    x[0] = range(0, 10)
    x[1] = range(3, 13)
    y[0] = range(0, 100, 10)
    y[1] = range(50, 150, 10)
    plt.plot(x[0], y[0],  label="0")
    plt.plot(x[1], y[1], label="1")
    plt.legend()
    plt.show()
