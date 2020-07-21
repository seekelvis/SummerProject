
from numpy import *
import matplotlib.pyplot as plt
import csv
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#从文件中加载数据：特征X，标签label
# CM: this function looks good. But in general it would be much easier to use Pandas dataframes - pd.read_csv()
# CM: you could then also impute the missing values using fillna() combined with df.mean()
# CM: also, for certain algorithms it is necessary to scale your data. See the use of StandardScaler in "Examples" here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
def loadDataSet(fileName):
    dataMatrix=[]
    dataLabel=[]

    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        m = 23 #特征数量 feature function number
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

# CM: Just a general comment - your functions model_SVM, model_LR and model_RF as do very similar things. You could think about how to reduce code duplciation...(I can make some suggestions if that helps)
# CM: Also, because it can take a long time to train models you can save the models using joblib so that you don't have to retrain everytime
def model_SVM(trainDataSet,trainLabelSet,testDataSet):
    # X = np.array(trainDataSet) #训练数据
    # Y = np.array(trainLabelSet) #训练标签
    # T = np.array(testDataSet) #预测数据
    X_train = trainDataSet
    Y_train = trainLabelSet
    # X_val = valDataSet
    # Y_val = valMatLabel
    T = testDataSet

    #gridsearch for find the best c
    best_score=0
    for c in [ 0.01, 0.1, 1]:
        # CM: I don't think this can be an infinite loop (that mainly happens with 'while')
        # CM: by adding a print statements you can see the loop progress. I suspect that it is just taking a long time.
        print(c)
        # 对于每种参数可能的组合，进行一次训练
        svc = SVC(C=c,kernel='linear',probability=True)
        # svc.fit(X_train,Y_train)
        scores = cross_val_score(svc, X_train,Y_train, cv=3)
        score = scores.mean()
        if score > best_score:
            best_score = score
            best_c = c

    svc=SVC(C=best_c,kernel='linear',probability=True) #注意函数参数说明
    svc.fit(X_train,Y_train)
    pre=svc.predict_proba(T)


    print('Best socre:{:.2f}'.format(best_score))
    print('Best parameters:{}'.format(best_c))
    return pre
# CM: you can also try to optimise the hyperparameters of the LR classifier...
def model_LR(trainDataSet,trainLabelSet,testDataSet):
    X = trainDataSet
    Y = trainLabelSet
    T = testDataSet
    # 逻辑回归模型
    log_model = LogisticRegression()
    # 训练逻辑回归模型
    log_model.fit(X, Y)
    # 预测y的值
    # pre = log_model.predict(T)
    pre = log_model.predict_proba(T)
    return pre

#RF model have some bugs, it can't work now
# CM: with Random forest there are many hyperparameters to optimise, so it is a good to use GridSearchCV
def model_RF(trainDataSet,trainLabelSet,testDataSet):
    # CM: it is not necessary to use tolist() here, it will accept numpy arrays
    X = trainDataSet.tolist()
    Y = trainLabelSet.tolist()
    T = testDataSet.tolist()
    # a.tolist()
    # 定义RF模型参数
    rf_model = RandomForestClassifier(n_estimators=200,max_depth=10,max_features='sqrt') # CM: 'sqrt' must be a string
    # 训练RF模型
    rf_model.fit(X, Y)
    # 预测值
    pre = rf_model.predict_proba(T)
    return pre

# CM: you could edit this function to accept multiple predictions and labels, these could be stored in a dictionary (which is very pythonic). For example....
# CM: predictions = {'test set (source)': (test_set_predictions, test_set_labels),
#                    'train set (source)': (train_set_predictions, train_set_labels),
#                    'test set (target)': (train_set_predictions, train_set_labels)}
# CM: This is just a suggestion, but the point is that you need to be able to compare the performance of a classifier on different data sets
# CM: similarly it will be useful to be able to compare the performance of different classifiers on the same dataset.
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

    # trainDataSet, trainLabelSet = loadDataSet('D:/code/SummerProject/shallowClassifier/data/train02.csv') #source domain
    # testDataSet, testMatLabel = loadDataSet('D:/code/SummerProject/shallowClassifier/data/test02.csv') #target domain

    # CM: I have changed the path to the csv file to be relative to the current directory (this way it is simpler, and also runs on my machine)
    dataSet, labelSet = loadDataSet('data/data01.csv') ##data01.csv is a file include source and target domain
    trainDataSet, testDataSet, trainLabelSet,  testMatLabel = model_selection.train_test_split(dataSet, labelSet, test_size=0.7, random_state=27) # CM: for now I just increased the size of the test set so that training is faster for the SVC. But you should just use MIMIC to train for the 'naive' approach
    # trainDataSet, valDataSet, trainLabelSet, valMatLabel = model_selection.train_test_split(trainDataSet, trainLabelSet, test_size=0.2, random_state=27)

    # sourceDataSet, sourceLabelSet = loadDataSet('D:/code/SummerProject/shallowClassifier/data/train01.csv')
    # targetData, targetLabel = loadDataSet('D:/code/SummerProject/shallowClassifier/data/test01.csv')
    # aData, testDataSet, aLabel, testMatLabel = model_selection.train_test_split(targetData, targetLabel, test_size=0.3, random_state=27)

    # tempData = sourceDataSet.tolist()
    # tempData.append(aData.tolist())
    # trainDataSet = np.array(tempData)
    # tempLabel = sourceLabelSet.tolist()
    # tempLabel.append(aLabel.tolist())
    # trainLabelSet = np.array(tempLabel)
    # trainDataSet = np.vstack(sourceDataSet,aData)
    # trainLabelSet = np.hstack(sourceLabelSet,aLabel)
    # print(sourceLabelSet)

    # #########   model_LR   ###########
    # LRpre = model_LR(trainDataSet, trainLabelSet, testDataSet)
    # analyze(LRpre, testMatLabel)

    ########   model_SVM   ###########
    # CM: I wouldn't worry too much at this stage, but it is worth knowing about PEP8 python stylistic convention
    # CM: it includes things like space after comma in function arguments (which I have added in the function call below)
    # CM: if you use an IDE (I use pyCharm) it will often help you by highlighting 'bad' styling and help you write more readable code.

    # print(trainDataSet.shape) # CM: with data01 and test_size=0.2 the training data has 17000 rows. This makes the SVM very slow to fit.
    # CM: one solution would be to use linearSVC (see here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    # CM: but for the 'naive' case that I suggested, you should train only on the MIMIC data. This will reduce the volume.
    #SVMpre = model_SVM(trainDataSet, trainLabelSet, testDataSet)
    #analyze(SVMpre, testMatLabel)

    #########   model_RF  ###########
    RFpre = model_RF(trainDataSet, trainLabelSet,testDataSet)
    analyze(RFpre,testMatLabel)
