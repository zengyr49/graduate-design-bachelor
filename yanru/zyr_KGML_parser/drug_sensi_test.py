from yanru.zyr_KGML_parser.process_of_analyzing_drug_sensitivity import analyzing_drug_sensitivity
import numpy as np


list_pearson=[]
list_esti=[]
analyzing_drug_sensitivity=analyzing_drug_sensitivity()
drugid=[133]
#note(drugid,n_estimator,epochs,learningrate)(133,17,200,0.01) (不专门记录了，基本上在n_estimate=20,epochs=200,lr=0.01的时候能够得到比较好的效果)

data_name='/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/'+'drug'+str(133)+'data.txt'
target_name='/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/'+'drug'+str(133)+'target.txt'

f_data=open(data_name,'r')
f_target=open(target_name,'r')
data=np.array(eval(f_data.read()))
target=np.array(eval(f_target.read()))

#加入pathway信息
print('now constructing pathway data...')
data_pathway=[]
indx_microarray=eval(open("/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/geneexpression_namelist.txt","r").read())
xml_path="/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/drug_target_pathway/hsa01524_TOP2.xml"
for data_vector in data:
    X,W=analyzing_drug_sensitivity.add_pathway_data(data_vector=data_vector,indx=indx_microarray,xml_path=xml_path)
    X=X.reshape(-1)
    W=W.reshape(-1)
    merge=np.array(np.hstack((X,W)))[0] #主要是matrix索引真麻烦
    data_pathway.append(merge)

data_pathway=np.array(data_pathway)
print('pathway data complete')
k_fold=5

for n_estimators in [20]:
    # analyzing_drug_sensitivity.feature_selection_by_randomforest(data=data,target=target,rf_n_estimator=n_estimators,drug_id=drugid)
    data_train, y_train, data_test, y_test = analyzing_drug_sensitivity.load_data(from_makeup_data=True,
                                                                                  isMicroarray=True, drug_id=drugid,for_pick_feature_test=True,test_data=data,test_target=target,data_split=5,save_info=0.66)
    data=np.vstack((data_train,data_test))
    data_and_pathway=np.hstack((data,data_pathway))

    #以下是交叉检验的步骤。如果出现out of bound 错误，则在test index end 前面加上判断语句
    for fold in range(k_fold):
        dims=len(data[0])
        eight_two = len(data) // 5
        train = len(data) - eight_two
        test_index_begin=eight_two*fold
        test_index_end=eight_two*(fold+1)


        data_test = data_and_pathway[test_index_begin:test_index_end]
        data_train = np.setdiff1d(data,data_test).reshape(-1,dims)

        pearson=analyzing_drug_sensitivity.zyr_mlp(data_train=data_train,data_test=data_test,target_test=y_test,target_train=y_train,isSave=False,drug_id=drugid,optimizer='sgd',epochs=200,learning_rate=0.01)
        list_pearson.append(pearson[0])
        list_esti.append(n_estimators)
        print(pearson)


maximum=max(list_pearson)
indx=list_pearson.index(maximum)
esi_best=list_esti[indx]

print(list_pearson)
print(list_esti)
print(maximum,esi_best)