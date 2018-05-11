import numpy as np
import math


class ROC:
    def __init__(self, result, label):
        self.result = result
        self.label = label
        self.sample_num = np.shape(result)[0]
        self.auc = 0
        assert self.sample_num == np.shape(label)[0]
        self.FPR_sorted, self.TPR_sorted = None, None
        self.computeROC()
        self.default_Sn, self.Sp, self.Acc, self.MCC = 0, 0, 0, 0
        # self.get_default_coef()

    def get_default_coef(self):
        self.default_Sn, FPR, self.Acc, self.MCC = self.default_coefs(thres=0.5)
        self.Sp = 1-FPR

    def default_coefs(self, thres):
        tmp_thres = thres
        tmp_pred_list = []
        for j in range(self.sample_num):
            if self.result[j] >= tmp_thres:
                tmp_pred_list.append(1)
            else:
                tmp_pred_list.append(0)
        tmp_TPR, tmp_FPR, Acc, MCC = self.getFourCoefs(pred_list=tmp_pred_list, label_list=self.label)
        return tmp_TPR, tmp_FPR, Acc, MCC


    def computeROC(self):
        FPR, TPR = [], []
        for i in range(self.sample_num):
            tmp_thres = self.result[i]
            tmp_pred_list = []
            for j in range(self.sample_num):
                if self.result[j] >= tmp_thres:
                    tmp_pred_list.append(1)
                else:
                    tmp_pred_list.append(0)
            tmp_TPR, tmp_FPR = self.getTPR_FPR(pred_list=tmp_pred_list, label_list=self.label)
            FPR.append(tmp_FPR)
            TPR.append(tmp_TPR)
        FPR_np = np.array(FPR)
        TPR_np = np.array(TPR)

        self.auc = self.getAUC(FPR=FPR_np, TPR=TPR_np)

    def getAUC(self, FPR, TPR):
        sort_index = np.argsort(FPR)
        # print(np.shape(sort_index))
        FPR_sorted = FPR[sort_index]
        self.FPR_sorted = FPR_sorted
        TPR_sorted = np.sort(TPR[sort_index])
        self.TPR_sorted = TPR_sorted
        Area = 0
        p_start, p_stop = 0, 0
        d_up, d_down = 0, 0
        for i in range(len(FPR_sorted)):
            p_stop = FPR_sorted[i]
            d_down = TPR_sorted[i]
            Area += (d_up + d_down)*(p_stop-p_start)*0.5
            p_start = p_stop
            d_up = d_down
        return Area

    def getTPR_FPR(self, pred_list, label_list):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(self.sample_num):
            if label_list[i] == 1:
                if pred_list[i] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_list[i] == 1:
                    FP += 1
                else:
                    TN += 1
        assert TP+FP+TN+FN == self.sample_num
        FPR = FP/(FP+TN)
        TPR = TP/(TP+FN)
        return TPR, FPR

    def getFourCoefs(self, pred_list, label_list):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(self.sample_num):
            if label_list[i] == 1:
                if pred_list[i] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_list[i] == 1:
                    FP += 1
                else:
                    TN += 1
        FPR = FP/(FP+TN)
        TPR = TP/(TP+FN)
        Acc = (TP+TN)/(TP+FN+TN+FP)
        MCC = ((TP*TN)-(FP*FN))/math.sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))
        return TPR, FPR, Acc, MCC
