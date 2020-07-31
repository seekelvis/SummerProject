import time

import joblib
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#从文件中加载数据：特征X，标签label
# CM: this function looks good. But in general it would be much easier to use Pandas dataframes - pd.read_csv()
# CM: you could then also impute the missing values using fillna() combined with df.mean()
# CM: also, for certain algorithms it is necessary to scale your data. See the use of StandardScaler in "Examples" here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
def load_data_set(fileName):
    # Read data
    data_feature = pd.read_csv(fileName, usecols=range(1,24), dtype=float)
    data_label = pd.read_csv(fileName, usecols=[25], dtype=int)
    # Fill missing data by means
    data_feature = data_feature.fillna(data_feature.mean())
    # Type Changing: dataFrame to array
    data_feature = data_feature.values
    data_label = data_label.values.reshape(-1,)

    # print(data_feature)
    # print(data_label)


    return data_feature , data_label

def model_build(classifier, trian_features, train_labels):
    if classifier=="LR":
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        param_grid = [{'logisticregression__C': [1, 10, 100]}]
    elif classifier=="SVM":
        # pipe = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)) SVC(kernel='linear',probability=True)
        pipe = make_pipeline(StandardScaler(), SVC(kernel='linear',probability=True))
        param_grid = [{'svc__C': [0.01, 0.1, 1]}]
    elif classifier=="RF":
        pipe = make_pipeline(StandardScaler(), RandomForestClassifier(max_features='sqrt'))
        param_grid = {'randomforestclassifier__n_estimators': range(10, 250, 10),
                      'randomforestclassifier__max_depth': range(5, 15, 1)
                      # 'learning_rate': np.linspace(0.01, 2, 20),
                      # 'subsample': np.linspace(0.7, 0.9, 20),
                      # 'colsample_bytree': np.linspace(0.5, 0.98, 10),
                      # 'min_child_weight': range(1, 9, 1)
        }
    model = GridSearchCV(pipe, param_grid, cv=3)
    model.fit(trian_features, train_labels)
    # save the model
    model_file_name = classifier + time.strftime("%m%d-%H%M%S") + ".model"
    joblib.dump(filename=model_file_name, value=model)
    return model

def model_predict(model, train_features_source, test_features_source, test_features_target):
    train_predictions_source = model.predict_proba(train_features_source)
    test_predictions_source = model.predict_proba(test_features_source)
    test_predictions_target = model.predict_proba(test_features_target)
    # return the probability of 1
    return train_predictions_source[:,1], test_predictions_source[:,1], test_predictions_target[:,1]

# CM: Just a general comment - your functions model_SVM, model_LR and model_RF as do very similar things. You could
# think about how to reduce code duplciation...(I can make some suggestions if that helps) CM: Also, because it can
# take a long time to train models you can save the models using joblib so that you don't have to retrain everytime
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
def analyze(predict_dictionary):

    fpr, tpr, thresholds = metrics.roc_curve(predict_dictionary['train_labels_source'],
                                             predict_dictionary['train_predictions_source'], pos_label=1)
    auc_train_source = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='train_source')

    fpr, tpr, thresholds = metrics.roc_curve(predict_dictionary['test_labels_source'],
                                             predict_dictionary['test_predictions_source'], pos_label=1)
    auc_test_source = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='test_source')

    fpr, tpr, thresholds = metrics.roc_curve(predict_dictionary['test_labels_target'],
                                             predict_dictionary['test_predictions_target'], pos_label=1)
    auc_test_target = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='test_target')

    plt.legend()
    plt.show()

    print("auc_train_source: ",auc_train_source)
    print("auc_test_source", auc_test_source)
    print("auc_test_target", auc_test_target)




    plt.show()
if __name__ == '__main__':
    # load data
    train_features_source, train_labels_source = load_data_set('data/Feature_Matrix_MIMIC_with_missing_values.csv')
    test_features_target, test_labels_target = load_data_set('data/Feature_Matrix_GICU_with_missing_values.csv')
    train_features_source, test_features_source, train_labels_source, test_labels_sourcel = model_selection.train_test_split(train_features_source, train_labels_source, test_size=0.7, random_state=27) # CM: for now I just increased the size of the test set so that training is faster for the SVC. But you should just use MIMIC to train for the 'naive' approach
    # initiate dict
    dict_origin = {'train_predictions_source': [], 'train_labels_source': train_labels_source,
                   'test_predictions_source': [], 'test_labels_source': test_labels_sourcel,
                   'test_predictions_target': [], 'test_labels_target': test_labels_target}
    # build a modle
    model_type_set = ["LR", "SVM", "RF"]
    model = model_build(model_type_set[2], train_features_source, train_labels_source)
    # prediction
    preciton_dict = dict_origin
    preciton_dict['train_predictions_source'], \
    preciton_dict['test_predictions_source'], \
    preciton_dict['test_predictions_target'] = model_predict(model, train_features_source, test_features_source, test_features_target)
    # analyze
    analyze(preciton_dict)


