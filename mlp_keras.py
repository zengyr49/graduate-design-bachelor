# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 18:56
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : mlp_keras.py
# @Software: PyCharm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD

x_all_train2 = "" #input data you want, type is like numpy 2D array
y_all_train1 = ""
x_all_test2 = ""


int_dim = len(x_all_train2[0])

model = Sequential()
model.add(Dense(int_dim // 2, input_dim=int_dim, init='uniform'))  # 输入层，17419，最好附加初始化，用uniform
# model.add(Dense(inputdim // 2, input_dim=inputdim, init='uniform',kernel_regularizer=regularizers.l2(0.0001)))

model.add(Activation('relu'))  # 激活函数是tanh(后面变成了relu因为对mnist的处理结果会好一些)
# model.add(Dropout(0.2))  # 采用50%的dropout

model.add(Dense(int_dim // 10))
model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Dense(int_dim // 100))
model.add(Activation('relu'))

model.add(Dense(int_dim // 1000))
model.add(Activation('relu'))

model.add(Dense(int_dim // 2000))
model.add(Activation('relu'))


model.add(Dense(1))  # 输出结果和药物对应的话如果只是二维的，结果会有one hot 编码
model.add(Activation('linear'))  # 最后一层sigmoid，因为是实际的结果

# 设定学习率（lr）等参数
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, verbose=0, mode='min', epsilon=0.0001,
#                               cooldown=0, min_lr=0)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
# check_test = ModelCheckpoint('best_model_test.h5', monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
hist = model.fit(x_all_train2, y_all_train1, batch_size=15, epochs=100, shuffle=True, verbose=2)
# model = load_model('best_model_test0518.h5')
predict = model.predict(x_all_test2)


