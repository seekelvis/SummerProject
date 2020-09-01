import time

import umap
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
from sklearn.manifold import TSNE
from libtlda.iw import ImportanceWeightedClassifier
from sklearn import preprocessing


# 从文件中加载数据：特征X，标签label
# CM: this function looks good. But in general it would be much easier to use Pandas dataframes - pd.read_csv()
# CM: you could then also impute the missing values using fillna() combined with df.mean()
# CM: also, for certain algorithms it is necessary to scale your data. See the use of StandardScaler in "Examples" here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
def load_data_set(fileName):
    # Read data
    data_feature = pd.read_csv(fileName, usecols=range(1, 24), dtype=float)
    data_label = pd.read_csv(fileName, usecols=[25], dtype=int)
    # Fill missing data by means
    data_feature = data_feature.fillna(data_feature.mean())
    # Type Changing: dataFrame to array
    data_feature = data_feature.values
    data_label = data_label.values.reshape(-1, )

    return data_feature, data_label


def model_build(classifier, trian_features, train_labels, test_features):
    if classifier == "IW":
        # pipe = make_pipeline(StandardScaler(), ImportanceWeightedClassifier(iwe='nn'))
        # param_grid = [{}]
        # model = GridSearchCV(pipe, param_grid, cv=3)
        # model.fit(trian_features, train_labels, test_features)
        # print(model.best_params_)


        model = ImportanceWeightedClassifier(iwe='kde')
        model.fit(preprocessing.scale(trian_features), train_labels, preprocessing.scale(test_features))


    else:
        if classifier == "LR":
            pipe = make_pipeline(StandardScaler(), LogisticRegression())
            param_grid = [{'logisticregression__C': [1, 10, 100]}]
        elif classifier == "SVM":
            # pipe = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)) SVC(kernel='linear',probability=True)
            pipe = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
            param_grid = [{'svc__C': [0.01, 0.1, 1]}]
        elif classifier == "RF":
            pipe = make_pipeline(StandardScaler(), RandomForestClassifier(max_features='sqrt'))
            param_grid = {'randomforestclassifier__n_estimators': range(230, 300, 10),
                          'randomforestclassifier__max_depth': range(8, 12, 1),
                          'randomforestclassifier__min_samples_leaf': range(1, 5, 1),
                          'randomforestclassifier__max_features': range(1, 20, 1)

                          # 'learning_rate': np.linspace(0.01, 2, 20),
                          # 'subsample': np.linspace(0.7, 0.9, 20),
                          # 'colsample_bytree': np.linspace(0.5, 0.98, 10),
                          # 'min_child_weight': range(1, 9, 1)
                          }
        model = GridSearchCV(pipe, param_grid, cv=3)
        model.fit(trian_features, train_labels)
        print(model.best_params_)
    # save the model
    model_file_name = classifier + time.strftime("%m%d-%H%M%S") + ".model"
    joblib.dump(filename=model_file_name, value=model)
    return model


def model_load(name):
    model1 = joblib.load(filename=name)
    return model1


def model_predict(model, train_features_source, test_features_source, test_features_target):
    train_predictions_source = model.predict_proba(train_features_source)
    test_predictions_source = model.predict_proba(test_features_source)
    test_predictions_target = model.predict_proba(test_features_target)
    # return the probability of 1
    print(test_predictions_target)
    return train_predictions_source[:, 1], test_predictions_source[:, 1], test_predictions_target[:, 1]


def tsne_show(x_features, y_features, x_labels, y_labels):
    tsne = TSNE(n_components=2)
    # print(type(x_features))
    # print(type(y_features))
    # print(np.shape(x_features))
    # print(np.shape(y_features))
    mix_features = x_features;
    mix_features = np.vstack((mix_features, y_features))
    # print(np.shape(mix_features))
    #
    # print(type(x_labels))
    # print(type(y_labels))
    # print(np.shape(x_labels))
    # print(np.shape(y_labels))
    mix_labels = [z + 2 for z in y_labels]
    mix_labels = np.append(x_labels, mix_labels)
    # print(np.shape(mix_labels))

    embedded = tsne.fit_transform(mix_features)
    colorset = ['r', 'g', 'purple', 'yellow']
    color_labels = [colorset[z] for z in mix_labels]
    plt.figure()
    plt.title("T-SNE")
    plt.scatter(embedded[:, 0], embedded[:, 1], s=0.8, c=color_labels)

    # range_begin = [0, 0, len(x_features), len(x_features)]
    # range_end = [len(x_features), len(x_features), -1,-1]
    # words_labels = ['source 0', 'source 1', 'target 0', 'target 1']
    # for i in range(4):
    #     plt.scatter(embedded[range_begin[i]:range_end[i], 0],
    #                 embedded[range_begin[i]:range_end[i], 1],
    #                 s=0.8, c=colorset[i], label=words_labels[i])

    # plt.legend(loc="best")
    fig1 = plt.gcf()
    fig1.show()


def umap_show(x_domain, y_domain):
    x_embedded = umap.UMAP().fit_transform(x_domain)
    y_embedded = umap.UMAP().fit_transform(y_domain)

    plt.figure()
    plt.title("source_domain_UMAP")
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
    fig1 = plt.gcf()
    fig1.show()

    plt.figure()
    plt.title("target_domain_UMAP")
    plt.scatter(y_embedded[:, 0], y_embedded[:, 1])
    fig2 = plt.gcf()
    fig2.show()


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

    print("auc_train_source: ", auc_train_source)
    print("auc_test_source", auc_test_source)
    print("auc_test_target", auc_test_target)



if __name__ == '__main__':
    # load data
    train_features_source, train_labels_source = load_data_set('data/Feature_Matrix_MIMIC_with_missing_values.csv')
    test_features_target, test_labels_target = load_data_set('data/Feature_Matrix_GICU_with_missing_values.csv')

    # train_features_source, test_features_source, train_labels_source, test_labels_source = \
    #     model_selection.train_test_split(train_features_source, train_labels_source, test_size=0.7, random_state=27)

    test_features_source, test_features_target, test_labels_source, test_labels_target = \
        model_selection.train_test_split(test_features_target, test_labels_target, test_size=0.7, random_state=27)

    # initiate dict
    dict_origin = {'train_predictions_source': [], 'train_labels_source': train_labels_source,
                   'test_predictions_source': [], 'test_labels_source': test_labels_source,
                   'test_predictions_target': [], 'test_labels_target': test_labels_target}

    # tsne_show(train_features_source, test_features_target, train_labels_source, test_labels_target)
    # umap_show(train_features_source, test_features_target)

    # build a modle
    model_type_set = ["LR", "SVM", "RF", "IW"]
    model = model_build(model_type_set[3], train_features_source, train_labels_source, test_features_source)

    # # prediction
    preciton_dict = dict_origin
    preciton_dict['train_predictions_source'], \
    preciton_dict['test_predictions_source'], \
    preciton_dict['test_predictions_target'] = model_predict(model, train_features_source, test_features_source,
                                                             test_features_target)

    # # analyze
    analyze(preciton_dict)
