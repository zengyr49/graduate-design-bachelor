import pretreat_data #TODO:注意了，这里的import最好用绝对路径来引入，因为pretreat_data是自己写的
import oldmethod_class #TODO:注意了，这里的import最好用绝对路径来引入，因为oldmethod_class也是自己写的，而且参数都已经弄好啦！比较适合挑选出来的特征
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from scipy import stats
from keras import regularizers
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
# AUC = metrics.roc_auc_score(y_true=,y_score=)
# fpr,tpr,thresholds = metrics.roc_curve(y_true=,y_score=,pos_label=) #横坐标为fpr（1-特异性），纵坐标为tpr（敏感性）
drug_id = 134

def load_and_normalized(drug_id=134,isMicroarray = True):
    #原本想把load RNAseq的也加进来的，后面发现好像意义不大，于是就舍弃了。到时候直接在process of analyzing drug sensitivity中load就行。
    # 前期主要在microarray data上做
    #使用134号药物作为测试吧！
    if isMicroarray == True:
        drug_id=drug_id
        database_path = "/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/"
        data_to_load = database_path + "drug" + str(drug_id) + "data.txt"
        target_to_load = database_path + "drug" + str(drug_id) + "target.txt"

        f_data = open(data_to_load,"r")
        f_target = open(target_to_load,"r")

        data = eval(f_data.read())
        target_str = eval(f_target.read())
        target = [float(i) for i in target_str]
        del target_str
        cut_avg = sum(target)/len(target)
        target_class = [] #这个是给机器学习用的；
        # target_sk = [] #这个是给sklearn一行用的；
        for i in target:
            if i >= cut_avg:
                target_class.append([1,0]) #大于均值，说明不敏感，说明是阴性，阴性用[1,0]表示；
                # target_sk.append(0)
            else:
                target_class.append([0,1]) #小于均值，敏感，阳性，用[0,1]表示；
                # target_sk.append(1)
        gapdh_exp = []
        for i in data:
            gapdh_exp.append(i[3756])
        gapdh_rev = 1 / np.array(gapdh_exp)
        data = np.array(data)
        for i, j in enumerate(data):
            data[i] = data[i] * gapdh_rev[i]
        target_class = np.array(target_class)
        data = np.array(data)

    return data,target_class




#version NMF
def esti_NMF(data,target_class,for_sklearn=True):
    data_out = pretreat_data.pretreat_by_nmf(data)
    if for_sklearn == True:
        target_sk = []
        for i in target_class:
            if i[1] == 1:
                target_sk.append(1)
            else:
                target_sk.append(0)
        target_sk = np.array(target_sk)
        return data_out,target_sk
    else:
        return data_out,target_class

#version PCA
def esti_PCA(data,target_class,for_sklearn=True):
    data_out = pretreat_data.pretreat_by_pca(data)
    if for_sklearn == True:
        target_sk = []
        for i in target_class:
            if i[1] == 1:
                target_sk.append(1)
            else:
                target_sk.append(0)
        target_sk = np.array(target_sk)
        return data_out, target_sk
    else:
        return data_out, target_class

#version RF
def esti_RF(data,target_class,for_sklearn=True,drug_id=134,save_and_load_path="/data/zengyanru/about_drug_sensitive/classification_mode/feature_ordinary_and_select_class/"):
    #save and load path 是保存feature的或者是读入feature文件的路径！
    from sklearn.ensemble import RandomForestClassifier
    target_sk = []
    for i in target_class:
        if i[1] == 1:
            target_sk.append(1)
        else:
            target_sk.append(0)
    pretreat_data.pretreat_by_randomforest(drug_id=134,data=data,target=target_sk)
    dataout = pretreat_data.load_rf_data(data_list=data,drug_id=134)
    if for_sklearn == True:
        return dataout,target_sk
    else:
        return dataout,target_class

#version PSO
def esti_PSO(data,target):
    pretreat_data.pretreat_by_pso(data_all=data,target_all=target)

#version GA
def esti_GA(data,target):
    pretreat_data.pretreat_by_ga(data_all=data,target_all=target)


def mlp(data_train,target_train,drug_id=drug_id,batch_size = 15,epochs = 100,isSave=False,save_in_path = ""):

    inputdim = len(data_train[0])
    model = Sequential()
    model.add(Dense(inputdim // 2, input_dim=inputdim, init='uniform',kernel_regularizer=regularizers.l2(0.0001)))  # 输入层，28*28=784，最好附加初始化，用identity
    model.add(Activation('relu'))  # 激活函数是tanh(后面变成了relu因为对mnist的处理结果会好一些)

    model.add(Dense(inputdim // 6, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l1(0.0001)))
    model.add(Activation('relu'))

    model.add(Dense(inputdim // 8, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l1(0.0001)))
    model.add(Activation('relu'))

    model.add(Dense(inputdim // 10, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l1(0.0001)))
    model.add(Activation('relu'))

    model.add(Dense(2))  # 输出结果和药物对应的话只是一维的
    model.add(Activation('sigmoid'))  # 最后一层linear，因为是实际的结果

    sgd = SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
    if isSave:
        modelname = save_in_path + 'model_mlp_fordrug' + str(drug_id) + '.h5'
        check_test = ModelCheckpoint(modelname, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False,
                                         mode='min', period=1)
        model.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                      callbacks=[check_test])
        # predicted = model.predict(data_test)
        # predicted = predicted.reshape(-1)
        # pearson = stats.pearsonr(predicted, target_test.reshape(-1))
        return model
    else:
        model.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
        # predicted = model.predict(data_test)
        # predicted = predicted.reshape(-1)
        # pearson = stats.pearsonr(predicted, target_test.reshape(-1))
        return model

if __name__ == '__main__':
    data,target_class = load_and_normalized(drug_id=134)
    data,target_class = esti_NMF(data=data,target_class=target_class,for_sklearn=True)

    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1234567)  # random state就是为了可重复而已
    sum_fpr = []
    sum_tpr = []
    sum_AUC = []

    for train,test in rkf.split(data):
        data_train,data_test,target_train,target_test = data[train],data[test],target_class[train],target_class[test]
        model = oldmethod_class.RandomForestC(data_train,target_train)
        y_pred = model.predict_proba(data_test)
        y_score = [i[1] for i in y_pred]
        fpr, tpr, thresholds = metrics.roc_curve(y_true=target_test, y_score=y_score, pos_label=1)  # pos_label一般都是1
        sum_fpr.append(fpr)
        sum_tpr.append(tpr)
        AUC = metrics.auc(y_true=target_test, y_score=y_score)
        sum_AUC.append(AUC)

    fpr_avg = sum(sum_fpr) / len(sum_fpr)  # 注意了，sum一个列表，如果其中再含np.array，最终的到的是array的相加，返回也是array。
    tpr_avg = sum(sum_tpr) / len(sum_tpr)
    AUC_avg = sum(sum_AUC) / len(sum_AUC)
    print(AUC_avg)

    #下面是对于mlp等神经网络的版本
    # data, target_class = load_and_normalized(drug_id=134)
    # data, target_class = esti_NMF(data=data, target_class=target_class, for_sklearn=False)
    #
    # rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1234567)  # random state就是为了可重复而已
    # sum_fpr = []
    # sum_tpr = []
    # sum_AUC = []
    #
    # target_auc = []
    # for i in target_class:
    #     if i[1] == 1:
    #         target_auc.append(1)
    #     else:
    #         target_auc.append(0)
    #
    #
    # for train, test in rkf.split(data):
    #     data_train, data_test, target_train, target_test,target_auc_test = data[train], data[test], target_class[train], target_class[
    #         test],target_auc[test]
    #     model = oldmethod_class.RandomForestC(data_train, target_train)
    #     y_pred = model.predict_proba(data_test)
    #     y_score = [i[1] for i in y_pred]
    #     fpr, tpr, thresholds = metrics.roc_curve(y_true=target_auc_test, y_score=y_score, pos_label=1)  # pos_label一般都是1
    #     sum_fpr.append(fpr)
    #     sum_tpr.append(tpr)
    #     AUC = metrics.auc(y_true=target_auc_test, y_score=y_score)
    #     sum_AUC.append(AUC)
    #
    # fpr_avg = sum(sum_fpr) / len(sum_fpr)  # 注意了，sum一个列表，如果其中再含np.array，最终的到的是array的相加，返回也是array。
    # tpr_avg = sum(sum_tpr) / len(sum_tpr)
    # AUC_avg = sum(sum_AUC) / len(sum_AUC)
    # print(AUC_avg)

features_list = eval(open("D:\zengyr\\about_drug_sensitivity\classification_mode\\test\\drugID134_features.txt","r").read())
genename = eval(open("D:\\zengyr\\drugsensitivity_rawdata\\geneexpression_namelist.txt","r").read())
fw = open("D:\zengyr\\about_drug_sensitivity\classification_mode\\test\\genes_for_GO.txt","w")

sum = 0
featureID_list = []  # del later in order to refresh
for j in features_list:
    if sum < 0.8:
        sum += j[0]
        featureID_list.append(j[1])  # 这里的格式是按照importance大小来排序的关于特征的index，存在列表里面
    else:
        pass


genenamelist = []
for i in featureID_list:
    genenamelist.append(genename[i])


for k in genenamelist:
    print(k,file=fw)



fw.close()