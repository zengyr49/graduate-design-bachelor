import numpy as np
def data_from_features(data,target,feature_list_path,info_percent=0.8,data_split=5):
    sums=0
    try:
        f_features = open(feature_list_path, "r")
        featurelist = eval(f_features.read())
    except:
        featurelist=np.loadtxt(feature_list_path)
    featureindex = []
    for i in featurelist:
        if sums < info_percent:
            sums += i[0]
            featureindex.append(i[1])
        else:
            pass
    featureindex_sorted = sorted(featureindex)  # featureindex是rf挑选出来的这种药物的重要性比较靠前的特征索引
    new_data=[]
    for j in data:
        new_data.append([j[idx] for idx in featureindex_sorted])
    new_data=np.array(new_data)
    eight_two = len(new_data) // data_split
    train = len(new_data) - eight_two
    data_train = np.array(new_data[:train])
    target_train = np.array(target[:train])
    data_test = np.array(new_data[train:])
    target_test = np.array(target[train:])
    return data_train,target_train,data_test,target_test

drug_target_pathway_file="/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/"+"drug_"+str(drug_id)+"_pathway.txt"
pathway_data=np.loadtxt(fname=drug_target_pathway_file)
drug_target_pathway_name="/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/feature_oridinary_and_list/drug"+str(drug_id)+"_pathway.txt"
analyzing_drug_sensitivity.feature_selection_by_randomforest(data=pathway_data,target=target,rf_n_estimator=n_estimators,drug_id=[drug_id],save_in_name=drug_target_pathway_name)
data_train, y_train, data_test, y_test=data_from_features(pathway_data,target,feature_list_path="/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/feature_oridinary_and_list/drug"+str(133)+"_pathway.txt")

pearson=analyzing_drug_sensitivity.zyr_mlp(data_train=data_train,data_test=data_test,target_test=y_test,target_train=y_train,isSave=False,drug_id=[drug_id],optimizer='sgd',epochs=50,learning_rate=0.01,kernel_regularizer=regularizers.l2(0.1))


