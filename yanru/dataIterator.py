import numpy as np


class DataIterator:
    def __init__(self, matrix_data, label_data, batchSize):
        self.isHasNext = True

        self.data = matrix_data
        self.label = label_data
        self.batchSize = batchSize
        self.tmp_batchSize = batchSize

        [self.sample_num, self.feat_dim] = np.shape(matrix_data)
        self.out_dim = np.shape(label_data)[1]

        self.StartIndex = -self.batchSize
        self.StopIndex = -1
        self.shuffle_data()

    def shuffle_data(self):
        self.tmp_batchSize = self.batchSize
        self.isHasNext = True
        index = np.arange(self.sample_num)
        np.random.shuffle(index) #下面几行非常冗余。实际上只需要data_new=self.data[np.random.shuffle(index)]。类似于R中的处理方式
        data_new = np.zeros([self.sample_num, self.feat_dim])
        label_new = np.zeros([self.sample_num, self.out_dim])
        for i in range(self.sample_num):
            data_new[i, :] = self.data[index[i], :]
            label_new[i, :] = self.label[index[i], :]
        self.data = data_new
        self.label = label_new
        self.StartIndex = -self.batchSize

    def next_batch(self):
        self.StartIndex += self.batchSize
        if self.StartIndex+self.batchSize-1 >= self.sample_num-1: #检查batch_size是否会大于数据集,原本只有！！大于号！！
            self.isHasNext = False
            self.StopIndex = self.sample_num-1
            self.tmp_batchSize = self.sample_num-self.StartIndex
        else:
            self.StopIndex = self.StartIndex+self.batchSize-1

        # print(self.tmp_batchSize)
        data_batch = np.zeros([self.tmp_batchSize, self.feat_dim])
        label_batch = np.zeros([self.tmp_batchSize, self.out_dim])

        # print('batch size:'+str(self.tmp_batchSize))
        # print('tmp start: '+str(self.StartIndex)+', stop:'+str(self.StopIndex))
        data_batch[:, :] = self.data[self.StartIndex:self.StopIndex+1, :]
        label_batch[:, :] = self.label[self.StartIndex:self.StopIndex+1, :]
        return data_batch, label_batch

    def data_inital(self):
        self.StartIndex = -self.batchSize
