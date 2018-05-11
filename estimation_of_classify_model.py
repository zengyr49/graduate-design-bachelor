# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:27
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : estimation_of_classify_model.py
# @Software: PyCharm

from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
import ml_models as oldmethod_class
from esti_pretreat_method import load_and_normalized
from esti_pretreat_method import esti_NMF

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

