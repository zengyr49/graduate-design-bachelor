# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:13
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : rf_pick_features.py
# @Software: PyCharm

def pretreat_by_randomforest(drugid,data,target,save_and_load_path="/data/zengyanru/about_drug_sensitive/classification_mode/feature_ordinary_and_select_class/",isMicroarray=True):
    #这个函数的到数据后需要自己再训练！
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    rf = RandomForestClassifier(oob_score=True, n_jobs=10, n_estimators=20,max_depth=13,max_features=300)
    # rf = RandomForestClassifier()
    writefilename = save_and_load_path + 'drugID' + str(drugid) + '_features.txt'
    fw_feature = open(writefilename, "w")
    data = np.array(data)
    target = np.array(target)
    names = [i for i in range(len(data[0]))]  # 这个就是索引，从0开始的
    rf.fit(data, target)
    print('now writing features to file...')
    print(sorted(zip(map(lambda x: round(x, 6), rf.feature_importances_), names), reverse=True),file=fw_feature)
    print('writing completed and begin the operation of next drug or exit')
    fw_feature.close()
    # del data
    # del target
    del rf

def load_rf_data(data_list,drug_id,feature_dir="/data/zengyanru/about_drug_sensitive/classification_mode/feature_ordinary_and_select_class/",pretreated_by_sd=False):
    import numpy as np
    print('now constructing feature index list...')
    if feature_dir == None:
        name_prefix = "/data/zengyanru/about_drug_sensitive/classification_mode/feature_ordinary_and_select_class/"
    else:
        name_prefix = feature_dir
    name_suffix = 'drugID' + str(drug_id) + '_features.txt'
    names = name_prefix + name_suffix
    f_features = open(names, 'r')  # close later in order to release memory
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
    # print(featureID_list_sorted)  #TODO:看sorted是不是按照数字来排序的！
    print('feature list has been finished')
    # 在此处准备读取进行机器学习的数据，并且用randomforest来做，保存214个模型
    print('now begin to load drug data...')
    print('now begin to normalize the data...')
    #下面就是normalize的步骤
    gapdh_exp = []
    for i in data_list:
        gapdh_exp.append(i[3756])
    gapdh_rev = 1 / np.array(gapdh_exp)
    data_list = np.array(data_list)
    for i, j in enumerate(data_list):
        data_list[i] = data_list[i] * gapdh_rev[i]
    if pretreated_by_sd == True:
        sd_dim_path = "/data/zengyanru/about_drug_sensitive/classification_mode/test/idxlist_by_sd.txt"
        idxlist = eval(open(sd_dim_path,"r").read())
        data_intermed = []
        for row in data_list:
            onerow = []
            for idx in idxlist:
                onerow.append(row[idx])
            data_intermed.append(onerow)
        data_list = data_intermed
    else:
        pass
    data = []
    # target = [float(l) for l in target_list]
    print('now compressing data dims...')
    for k in data_list:
        data_oneelement = [k[idx] for idx in featureID_list_sorted]
        data.append(data_oneelement)
    return data

