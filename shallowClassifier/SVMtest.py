# import numpy as np
# from sklearn.svm import SVC
#
# def SVM(trainDataMatrix,trainMatLabell,testDataMatrix):
#
#     X = np.array(trainDataMatrix) #训练数据
#     Y = np.array(trainMatLabell) #训练标签
#     T = np.array(testDataMatrix) #预测数据
#     print(X.shape,Y.shape)
#     svc=SVC(C=0.1,kernel='linear') #注意函数参数说明
#     svc.fit(X,Y)
#     pre=svc.predict(T)
#     print(X.shape,Y.shape)
#     print(pre)
#     return pre

#
# from numpy import *
# import matplotlib.pyplot as plt
# import csv
# from sklearn import metrics
# from sklearn.metrics import auc
# import numpy as np
# from sklearn.svm import SVC
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# import numpy as np
# X = np.random.random((10,5)) #训练数据
# Y = np.array([1,0,1,0,1,0,1,0,1,0]) #训练标签
# T = np.random.random((20,5)) #预测数据
# TY = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
# print(X.shape,Y.shape)
# svc=SVC(kernel='poly',degree=3,gamma=1,coef0=0) #注意函数参数说明
# svc.fit(X,Y)
# pre=svc.predict(T)
# print(X.shape,Y.shape)
# print(pre)

import numpy as np
# from sklearn import metrics
# y = np.array([0, 0, 1, 1])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
# # metrics.auc(fpr, tpr)
# #
# # fpr, tpr, thresholds = metrics.roc_curve(TY, pre, pos_label=1)
# auc = metrics.auc(fpr, tpr)
# print("fpr")
# print(fpr)
# print("tpr")
# print(tpr)
# print("thresholds")
# print(thresholds)
# print("auc")
# print(auc)

# sourceDataMatrix =np.array([[1,2,3],[4,5,6],[7,8,9]])
# aData = np.array([10,11,12])
# print(sourceDataMatrix)
# print(aData)
#
# tempData = sourceDataMatrix.tolist()
# print(tempData)
# tempData.append(aData.tolist())
# print(tempData)
# trainDataMatrix = np.array(tempData)
# print(trainDataMatrix)
# # tempLabel = sourceMatLabell.tolist()
# # tempLabel.append(aLabel.tolist())
# # trainMatLabell = np.array(tempLabel)

sourceDataMatrix =np.array([[1,2,3],[4,5,6],[7,8,9]])
a = sourceDataMatrix.shape()
print(a[0])