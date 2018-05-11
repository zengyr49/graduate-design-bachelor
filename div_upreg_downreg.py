# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:22
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : div_upreg_downreg.py
# @Software: PyCharm

#target_sk是细胞系对于药物的敏感性列表。是通过ic50值的平均数判断的。
#target_sk records whether cell line is sensitive to exact drug. it is divided by average of ic50.
#data_rf is data pretreat by standard devious and randomforest for eliminating some of the unwanted features

import numpy as np

sensi = []
anti = []
for i,j in enumerate(target_sk):
    if j == 1:
        sensi.append(data_rf[i])
    else:
        anti.append(data_rf[i])
sensi = np.array(sensi)
anti = np.array(anti)
genenamelist = eval(open("geneexpression_namelist.txt","r").read())
idxlist = eval(open("idxlist_by_sd.txt","r").read())
genenamesd = [genenamelist[i] for i in idxlist]
f_features = open("drugID134_features.txt", 'r')  # close later in order to release memory
features_list = eval(f_features.read())
f_features.close()
sum = 0
featureID_list = []  # del later in order to refresh
for j in features_list:
    if sum < 0.8:
        sum += j[0]
        featureID_list.append(j[1])  # 这里的格式是按照importance大小来排序的关于特征的index，存在列表里面
    else:
        pass
featureID_list_sorted = sorted(featureID_list)
genes = [genenamesd[idx] for idx in featureID_list_sorted]
sensi_t = np.transpose(sensi)
anti_t = np.transpose(anti)
from scipy.stats import ttest_ind
fw_ttest=open("ttest_gene_sensi_or_not.txt","w")
for i,j in enumerate(sensi_t):
    t,p = ttest_ind(j,anti_t[i],equal_var=False)
    gene = genes[i]
    print("%s\t%f\t%f" % (gene,t,p),file=fw_ttest)
fw_ttest.close()

