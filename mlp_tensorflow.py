# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:07
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : mlp_tensorflow.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from scipy import stats
import gzip
import six.moves.cPickle as pickle
from yanru import input_data
import os
from sklearn.preprocessing import OneHotEncoder
from yanru.dataIterator import DataIterator


print('now loading datasets...')
dataset='D:\zengyr\mnist'
dataset=dataset+os.sep
# print(dataset)
# mnist = input_data.read_data_sets(dataset,one_hot=True)


#尝试另一种读取数据的方法
dataset=dataset+'mnist.pkl.gz'
with gzip.open(dataset, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)
enc=OneHotEncoder()
y_train=enc.fit_transform(train_set[1].reshape(-1,1)).toarray()
y_valid=enc.fit_transform(valid_set[1].reshape(-1,1)).toarray()
y_test=enc.fit_transform(test_set[1].reshape(-1,1)).toarray()
# print(y[1:100])
# print(y_test[1:20])


#########尝试实现MLP，非师兄版本###########
lr=0.01
epochs=100
batch_size=100
display_step=1 #为了保证在有epoch的时候显示结果的，不用于mlp的计算，仅仅用于显示结果，不需要更改

n_hidden_1=392
n_hidden_2=196
n_input=784
n_classes=10

x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])

def mlp(x,W,b):
    layer_1=tf.add(tf.matmul(x,W['h1']),b['b1'])
    layer_1=tf.nn.relu(layer_1)
    layer_2=tf.add(tf.matmul(layer_1,W['h2']),b['b2'])
    layer_2=tf.nn.relu(layer_2)
    out_layer=tf.matmul(layer_2,W['out'])+b['out']
    return out_layer

W={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],seed=123)),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
b={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

pred=mlp(x,W,b)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)



init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

print(sess.run(W['h2']))

data_train=DataIterator(matrix_data=train_set[0],label_data=y_train,batchSize=batch_size)
print(train_set[0].shape)
num_example=len(train_set[0])
for eps in range(epochs):
    avg_cost=0
    # print(data_train.StartIndex)
    while data_train.isHasNext:
        batch_x,batch_y=data_train.next_batch()
        ops,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        avg_cost+=c/(int(num_example/batch_size))
    if eps % display_step==0:
        print('Epoch:','%04d'%(eps+1),'cost=','{:.9f}'.format(avg_cost))
print('Optimization complete')

correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype='float'))
print(sess.run(accuracy, feed_dict={x: test_set[0], y: y_test}))


print(sess.run(W['h2']))


#以下是保存模型
saver=tf.train.Saver(max_to_keep=1)
saver.save(sess,'ckpt\mnist.ckpt')
del sess
sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,'ckpt\mnist.ckpt')

print(sess.run(accuracy, feed_dict={x: test_set[0], y: y_test}))

