import pickle
from PIL import Image
import os
from feature import NPDFeature
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.

        '''
        #初始化
        self.weak_classifier=weak_classifier#弱分类器
        self.n_weakers_limit=n_weakers_limit#弱分类器个数
        self.strong_learner = []#强分类器
        self.alpha = []#分类器权重

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X_train,y_train):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        #初始化权重为1/样本数
        weight = np.array([float(1) / X_train.shape[0]] * X_train.shape[0])
        y_train=y_train.reshape((y_train.shape[0]))
        function_param = []
        mode_list=[]
        i=0;
        while i<self.n_weakers_limit:
            #基分类器训练
            mode = self.weak_classifier
            mode.fit(X_train, y_train, sample_weight=weight)
            train_predict = mode.predict(X_train)

            I_error = weight[train_predict- y_train != 0].sum()
            if I_error > 0.5:
                break
            if (I_error < 0.000000001):
                I_error = 0.000000001
            #误差率
            alpha=0.5 * np.log((1 - I_error) / I_error)
            Z=weight*np.exp(-1*alpha*y_train*train_predict)
            #更新权重
            weight = Z / Z.sum()
            #添加基分类器，组成强分类器
            function_param.append(alpha)
            mode_list.append(mode)
            i+=1
        self.strong_learner=mode_list
        self.alpha = function_param


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        i=0
        y=np.zeros((X.shape[0],1))
        #多个弱分类器组成强分类器
        while i<len(self.strong_learner):
            mode=self.strong_learner[i]
            param=self.alpha[i]
            predict=mode.predict(X)
            predict=np.array(predict).reshape(len(predict),1)
            y=y+param*predict
            i+=1;
        return y

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y=np.zeros((X.shape[0],1))
        #获得预测权重
        H=self.predict_scores(X)
        #sign的方法，得到预测值
        y[np.where(H>threshold)]=1
        y[np.where(H<=threshold)]=-1
        return y

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
