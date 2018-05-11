import numpy as np
from yanru.pso_yanru import pso
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy
from sklearn.externals import joblib
import random
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

print('Now,loading ic50 data...')

#numpy.set_printoptions(threshold=numpy.nan)
fr1=open('CNVdata_fixed_train_origin.txt','r')
fr2=open('CNVdata_fixed_test_origin.txt','r')
data_all=eval(fr1.read())
test_all=eval(fr2.read())

print('Origin data and target complete. Now begin to calculate.')

x_all_train1 = [i[0] for i in data_all]
y_all_train1 = [i[1] for i in data_all]
x_all_test1 = [i[0] for i in test_all]
y_all_test1 = [i[1] for i in test_all]

positive_all = y_all_test1.count(1)
negative_all = y_all_test1.count(0)

x_all_train1 = numpy.array(x_all_train1)
y_all_train1 = numpy.array(y_all_train1)
x_all_test1 = numpy.array(x_all_test1)
y_all_test1 = numpy.array(y_all_test1)

y_all_test1 = y_all_test1.tolist()
model = Sequential()
model.add(Dense(900, input_dim=46221, init='uniform'))  # 输入层，28*28=784，最好附加初始化，用identity
model.add(Activation('relu'))  # 激活函数是tanh(后面变成了relu因为对mnist的处理结果会好一些)
# model.add(Dropout(0.2))  # 采用50%的dropout

model.add(Dense(500))  # 隐层节点500个
model.add(Activation('relu'))
# model.add(Dropout(0.2))


model.add(Dense(1))  # 输出结果和药物对应的话只是一维的
model.add(Activation('sigmoid'))  # 最后一层linear，因为是实际的结果

# 设定学习率（lr）等参数
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, verbose=0, mode='min', epsilon=0.0001,
                              cooldown=0, min_lr=0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
#check_test = ModelCheckpoint('best_model_test0518.h5', monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
hist = model.fit(x_all_train1, y_all_train1, batch_size=100, nb_epoch=1, shuffle=True, verbose=2,
                 callbacks=[reduce_lr])
#model = load_model('best_model_test0518.h5')
predict_proba = model.predict_proba(x_all_test1)
content=(predict_proba, y_all_test1)
fw1=open('try.txt','w')
print(content,file=fw1)

cutoff = []
for i in range(1002):
    cutoff.append(0.001 * i)
sumlist = []
# fw = open('MLP_threshold_result_for_ROC_test.txt', 'w')
result_Sn = []
result_Sp = []
proba = content[0]
y_test = content[1]
for threshold in cutoff:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    Sn = 0
    Sp = 0

    for i in proba:
        index = proba.index(i)
        if i[0] >= threshold and y_test[index] == 1:
            TP += 1
            # elif i[0] >= threshold and y_test[index] == 0:
            # FP += 1
            # elif i[0] < threshold and y_test[index] == 1:
            # FN += 1
        elif i[0] < threshold and y_test[index] == 0:
            TN += 1
    '''if TP+FN==0 or TN+FP==0:
                TP=0
                FN=100
                TN=0
                FP=100'''
    Sn = TP / positive_all
    Sp = TN / negative_all
    result_Sn.append(Sn)
    result_Sp.append(Sp)
result_oneminusSp = []
for m in result_Sp:
    result_oneminusSp.append(1 - m)
# print(result_oneminusSp, file=fw)
# print(result_Sn, file=fw)
sum = 0
for n in range(len(result_Sn) - 1):
    x1 = result_Sn[n]
    x2 = result_Sn[n + 1]
    h = result_oneminusSp[n] - result_oneminusSp[n + 1]
    area = (x1 + x2) * h / 2
    sum += area
print(sum)





#define the function: maybe the NN. and the x would be the weight of characters in all dims
'''def weight(x,*args):#TODO:actually, x is what we want!
    all_dimentions=args  #TODO:remenber to transfer the data into array
    creatvar=locals()
    for i in range(len(args[1])):
        creatvar['w'+str(i)]=x[i]
    data_for_train=all_dimentions*x
    #TODO:add an algorithm that applys to our data(all_dimensions), actually mlp'''


