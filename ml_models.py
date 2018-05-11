# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:08
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : ml_models.py
# @Software: PyCharm

# coding = utf-8
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

kf = KFold(n_splits=10)

def linearReg(data,target):
    from sklearn import linear_model
    linear=linear_model.LinearRegression()
    data=np.array(data)
    target=np.array(target)
    for train,test in kf.split(data):
        data=np.array(data)
        target=np.array(target)
        x_train, x_test, y_train, y_test=data[train], data[test], target[train], target[test]
        linear.fit(x_train, y_train)
        predicted = linear.predict(x_test)
        mse = mean_squared_error(predicted, y_test)
        spearman_correlation = stats.spearmanr(predicted, y_test)
        print('LinearReg:\n', spearman_correlation, 'mse=%s' % mse)
    return linear


def LogisticReg(data,target):
    from sklearn .linear_model import LinearRegression
    from scipy import stats
    model=LinearRegression()
    for train,test in kf.split(data):
        data=np.array(data)
        target=np.array(target)
        x_train, x_test, y_train, y_test=data[train], data[test], target[train], target[test]
        model.fit(x_train,y_train)
        model.score(x_train,y_train)
        predicted=model.predict(x_test)

        mse=mean_squared_error(predicted,y_test)
        spearman_correlation=stats.spearmanr(predicted,y_test)
        print('LogReg:\n',spearman_correlation,'mse=%s'%mse)
    return model


def RandomForestC(data, target):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from scipy import stats
    model = RandomForestClassifier(n_estimators=20,max_depth=12,n_jobs=10)
    # data = np.array(data)
    # target = np.array(target)
    model.fit(data, target)
    # predicted=model.predict_proba(x_test)
    return model


def RandomForestR(data, target):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from scipy import stats
    model = RandomForestRegressor(n_estimators=100)
    for train, test in kf.split(data):
        data = np.array(data)
        target = np.array(target)
        x_train, x_test, y_train, y_test = data[train], data[test], target[train], target[test]
        y_train = y_train.ravel()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        '''print('Predict y:',predicted)'''
        spearman_correlation = stats.spearmanr(predicted, y_test)
        mse = mean_squared_error(predicted, y_test)
        print('RFR:\n', spearman_correlation, 'mse=%s' % mse)
    return model


def SVMC(data, target):
    from sklearn import svm
    from scipy import stats
    import numpy as np
    model = svm.SVC()
    model.fit(data,target)
    # for train, test in kf.split(data):
    #     data = np.array(data)
    #     target = np.array(target)
    #     x_train, x_test, y_train, y_test = data[train], data[test], target[train], target[test]
    #     y_train = y_train.ravel()
    #     model.fit(x_train, y_train)
    #     model.score(x_train, y_train)
    #     predicted = model.predict(x_test)
    #     '''print('Predict y:',predicted)'''
    #     spearman_correlation = stats.spearmanr(predicted, y_test)
    #     mse = mean_squared_error(predicted, y_test)
    #     print('SVMC:\n', spearman_correlation, 'mse=%s' % mse)
    return model


def SVMR(data, target):
    from sklearn import svm
    from scipy import stats
    import numpy as np
    model = svm.SVR()
    for train, test in kf.split(data):
        data = np.array(data)
        target = np.array(target)
        x_train, x_test, y_train, y_test = data[train], data[test], target[train], target[test]
        y_train = y_train.ravel()
        model.fit(x_train, y_train)
        model.score(x_train, y_train)
        predicted = model.predict(x_test)
        '''print('Predict y:',predicted)'''
        spearman_correlation = stats.spearmanr(predicted, y_test)
        mse = mean_squared_error(predicted, y_test)
        print('SVMR:\n', spearman_correlation, 'mse=%s' % mse)
    return model


def GBCT(data, target):
    from sklearn.ensemble import GradientBoostingClassifier
    from scipy import stats
    import numpy as np
    model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.001, max_depth=12, random_state=1238823,)
    model.fit(data,target)
    # for train, test in kf.split(data):
    #     data = np.array(data)
    #     target = np.array(target)
    #     x_train, x_test, y_train, y_test = data[train], data[test], target[train], target[test]
    #     y_train = y_train.ravel()
    #     model.fit(x_train, y_train)
    #     model.score(x_train, y_train)
    #     predicted = model.predict(x_test)
    #     Scores = model.score(x_test, y_test)
    #     '''print('Predict y:',predicted)'''
    #     spearman_correlation = stats.spearmanr(predicted, y_test)
    #     mse = mean_squared_error(predicted, y_test)
    #     print('GBCT:\n', spearman_correlation, 'mse=%s' % mse)
    return model


def GBRT(data, target):
    from sklearn.ensemble import GradientBoostingRegressor
    from scipy import stats
    import numpy as np
    from sklearn.metrics import mean_squared_error
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    for train, test in kf.split(data):
        data = np.array(data)
        target = np.array(target)
        x_train, x_test, y_train, y_test = data[train], data[test], target[train], target[test]
        y_train = y_train.ravel()
        model.fit(x_train, y_train)
        model.score(x_train, y_train)
        predicted = model.predict(x_test)
        Scores = model.score(x_test, y_test)
        '''print('Predict y:',predicted)
       print('Scores is:',Scores)
        mse=mean_squared_error(y_test,predicted)
       print('Mean squared error is:',mse)'''
        spearman_correlation = stats.spearmanr(predicted, y_test)
        mse = mean_squared_error(predicted, y_test)
        print('GBRT:\n', spearman_correlation, 'mse=%s' % mse)
    return model


