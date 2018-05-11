import numpy as np

class analyzing_drug_sensitivity:
    def __init__(self,drug_pick=[133,134,135,179,1005,1007,1114,1199,1375],path_to_save_feature_list=None):
        self.drug_pick = drug_pick
        if path_to_save_feature_list==None:
            path_to_save_feature_list='/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/feature_oridinary_and_list/' #左老师25 的是："/data1/data/zengyanru/feature_oridinary_and_list/"
        self.path_to_save_feature_list=path_to_save_feature_list

    def feature_selection_by_randomforest(self,data, target, drug_id, name_prefix=None,rf_n_estimator=20,n_jobs=15,save_in_name=None):
        #这是一个用随机森林挑选特征的函数，通过调整参数可以获得不同的特征
        #因为这是专门给我自己用的，而且药物就只有那么几种，因此我直接把药物的列表也放上来好了
        #name_prefix是药物数据所在的文件夹绝对路径,也是存放features的绝对路径
        #data_matrix,lable_matrix就顾名思义的
        #drug_id一定要是一个列表，存储着drug id，数字。TODO：必须和load_data的列表一致！！
        #n_jobs是rf跑起来时候的线程数目
        #save in name一定要是：drugID药物id_features.txt这种格式
        import numpy
        from sklearn.ensemble import RandomForestRegressor
        from sklearn import preprocessing

        '''drug_pick = [133, 134, 135, 136, 140, 147, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 163, 164, 165, 166,
                     167, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 184, 185, 186, 190, 192, 193, 194, 196,
                     197, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 211, 219, 221, 222, 223, 224, 225, 226, 228,
                     229, 230, 231, 235, 238, 245, 249, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263, 265, 266,
                     268, 269, 271, 272, 273, 274, 275, 276, 277, 279, 281, 282, 283, 286, 287, 288, 290, 291, 292, 293,
                     294, 295, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 312, 326, 328, 329, 330, 331,
                     332, 333, 341, 344, 345, 346, 1001, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
                     1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1026, 1028, 1029, 1030,
                     1031, 1032, 1033, 1036, 1037, 1038, 1039, 1042, 1043, 1047, 1049, 1050, 1052, 1053, 1054, 1057,
                     1058, 1059, 1060, 1061, 1062, 1066, 1067, 1069, 1072, 1091, 1114, 1129, 1133, 1149, 1170, 1175,
                     1192, 1194, 1199, 1218, 1219, 1230, 1236, 1239, 1241, 1242, 1243, 1248, 1259, 1261, 1262, 1264,
                     1268, 1371, 1372, 1373, 1375, 1377, 1378, 1494, 1495, 1498, 1502, 1526, 1527]'''
        if drug_id==None:
            drug_pick=[133,134,135,179,1005,1007,1114,1199,1375] #比较多病人数据的9种药物
        else:
            drug_pick=drug_id
        # print(len(drug_pick))  drug_pick中的个数一定要是214
        if name_prefix==None:
            nameprefix = '/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/'
        else:
            nameprefix=name_prefix
        if len(drug_pick) == len(drug_pick):
            for i in drug_pick:
                print('now loading ic50data for drug', i)

                rf = RandomForestRegressor(oob_score=True, n_jobs=n_jobs,n_estimators=rf_n_estimator,random_state=1234567)
                if save_in_name==None:
                    writefilename = self.path_to_save_feature_list+'drugID' + str(i) + '_features.txt'  # drugIDXX_features.txt
                    fw_feature = open(writefilename, 'w')
                else:
                    writefilename=save_in_name
                    fw_feature=open(writefilename,'w')

                data = numpy.array(data)
                target = numpy.array(target)

                print('drug', i, 'data has been successfully established and begin to pick out features...')

                names = [i for i in range(len(data[0]))]  # 这个就是索引，从0开始的
                rf.fit(data, target)

                print('now writing features to file...')

                print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True),
                      file=fw_feature)

                print('writing completed and begin the operation of next drug or exit')

                fw_feature.close()
                # del data
                # del target
                del rf
        else:
            print('error,please check the drug_pick and rerun.')

    def add_pathway_data(self,data_vector,indx,xml_path,gene_exp_path=None,pathway_type='genes',restart_prob=0.5,max_iter=100,precision=0.0000001,first_d=10,save_genename_list_path=None):
        from yanru.zyr_KGML_parser import RWR_and_context_node
        from yanru.zyr_KGML_parser.RWR_and_context_node import context_node_feature_SVDversion
        from yanru.zyr_KGML_parser.construct_matrix import construct_matrix_for_enzymes_RWR
        from yanru.zyr_KGML_parser.construct_matrix import construct_matrix_for_gene_RWR
        from yanru.zyr_KGML_parser.construct_matrix import construct_geneid_dict
        from yanru.zyr_KGML_parser.construct_matrix import construct_data_matrix
        from yanru.zyr_KGML_parser.construct_matrix import construct_id_name_map
        from yanru.zyr_KGML_parser.construct_matrix import construct_relation_dict
        from yanru.zyr_KGML_parser.construct_matrix import construct_reaction_dict
        from yanru.zyr_KGML_parser.construct_matrix import find_corresp_geneexp

        #加入pathway信息。pathway经过RWR，SVD 处理。最终返回一个矩阵吧。
        #xml_path 是pathway的信息。是整个文件的绝对路径
        #gene_exp_path 是基因表达的信息，必须是下载时候原始的那种类型，就是类似于R能够读取的那种类型。就是cancerrx上原始数据的模样
        #pathway_type 可选有'genes', 'enzymes'
        #其余参数都是给RWR的，可以参见下方关于RWR的函数
        #TODO:如果是RNA-seq数据，则在外部脚本中加入data, indx = construct_data_matrix(gene_exp_path)，然后再遍历data，加入特征
        #data_vetor是data_matrix中某一个病人的表达数据; indx 是基因名字的列表，形如：[基因名，基因名，基因名]。顺序必须与data_vector一致
        #TODO：如果是microarray数据，则在外部脚本遍历加载的病人基因，然后加到挑选过的特征后方
        #indx是

        genedict = construct_geneid_dict(xml_path, data_vector, genename_index=indx)

        geneiddict = construct_id_name_map(xml_path)

        relation_dict = construct_relation_dict(xml_path)

        if pathway_type=='enzymes':
            reaction_dict = construct_reaction_dict(xml_path)
            enzyme_matrix = construct_matrix_for_enzymes_RWR(genedict, geneiddict, relation_dict, reaction_dict,save_genename_list_path=save_genename_list_path)
            matrix_for_RWR=RWR_and_context_node.RWR(matrix=enzyme_matrix,restart_prob=restart_prob,max_iter=max_iter,precision=precision)
            final_matrix=context_node_feature_SVDversion(matrix_for_approaching=matrix_for_RWR,first_d=first_d)
        elif pathway_type=='genes':
            gene_matrix=construct_matrix_for_gene_RWR(genedict,geneiddict,relation_dict,save_genename_list_path=save_genename_list_path)
            matrix_for_RWR = RWR_and_context_node.RWR(matrix=gene_matrix, restart_prob=restart_prob,
                                                      max_iter=max_iter, precision=precision)
            final_matrix = context_node_feature_SVDversion(matrix_for_approaching=matrix_for_RWR, first_d=first_d)
        else:
            print('your pathway type must be either genes or enzymes!')
            raise NameError

        return final_matrix


    def load_data(self,drug_id,genenamelist_path=None,patient_followup_name=None,cancerexp_tcga=None,feature_dir=None,from_download_data=False,from_makeup_data=True,isMicroarray=True,microarray_dir=None,for_pick_feature_test=False,test_data=None,test_target=None,data_split=5,save_info=0.8):
        #用于载入数据，最好有选择的那种。因为载入数据有可能是从原始的下载的数据中载入，也可能是从整理过的数据中载入的.
        #一般都用makeup data吧！
        #如果是用raw data 的话，那么就不将输入的信息用于处理什么了。因为没有挑选步骤。或者干脆删除，平时不使用即可
        #这里录入的数据都是经过标准化的。目前选择的标准化使用GAPDH来做。因为z-score（正态分布那种）标准化的效果已经尝试过，并不好
        #cancerexp_tcga需要是一个文件夹路径，基本不用修改，因为它在服务器上，是TCGA上各种癌症的RNAseq基因表达数据；
        # patient_followup_name是具体的文件名字，genenamelist_path也是具体文件名字，是基因名字的列表，已经做好了的，17419个基因的。
        # 名字里带有dir的都是文件夹路径
        #drug_id一定要是list,TODO:这里的drug_id一定要和feature_selection_by_randomforest的一致！！
        #load_data没写好。照理说它应该只返回一个药物的信息的。列表改一下吧！
        #TODO：特别注意，load_data在不同的服务器上用不同的路径。另外，feature_list要自己去搬不同的服务器
        import numpy as np

        if from_makeup_data==True and isMicroarray==False and for_pick_feature_test==False:
            import re
            import numpy as np

            ##TODO：将一些中间变量删除避免内存溢出；添加指示语句以debug
            ##36个癌种的情况，基因表达在癌种1文件中，共36个
            if drug_id==None:
                druglist_pickout=[134]
                #druglist_pickout = [133, 134, 135, 179, 1005, 1007, 1114, 1199, 1375]  # 都是预测性能>0.618，而且病人用药样本数超过100的，只有9个，因此用枚举法。
            else:
                druglist_pickout=drug_id
            if genenamelist_path==None:
                try:
                    f_genenamelist = open("/data1/data/zengyanru/findout_followups/geneexpression_namelist.txt", "r")  # TODO：记得改成绝对路径或者可以外部引入的路径
                except:
                    f_genenamelist=open("/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/geneexpression_namelist.txt","r")
            else:
                f_genenamelist = open(genenamelist_path, "r")
            genenamelist = eval(f_genenamelist.read())

            # 字典的构造需要改变一下
            print('now establishing genedict...i need much memory for example 30G!!')
            cancerlist = ['UVM', 'UCS', 'UCEC', 'THYM', 'THCA', 'TGCT', 'STAD', 'SKCM', 'SARC', 'READ', 'PRAD', 'PCPG',
                          'PAAD', 'OV', 'MESO', 'LUSC', 'LUAD', 'LIHC', 'LGG', 'LAML', 'KIRP', 'KIRC', 'KICH', 'HNSC',
                          'GBM', 'ESCA', 'DLBC', 'COAD', 'CHOL', 'CESC', 'BRCA', 'BLCA', 'ACC']

            for id in druglist_pickout:
                missing_value = []
                cancerexpdict = {}
                notmatch_genelist = []
                patient_list = []
                patient_list_allsample = []  # 因为有的病人样本(组织样本也就是TCGA数据中的样本)可能大于1个，因此需要记录下来，在predict的时候作为顺序参考
                ###建立三个文件，纠错###
                notmatch_patient = "notmatch_patient_drug" + str(id) + ".txt"
                notmatch_gene = "notmatch_gene_drug" + str(id) + ".txt"
                predict_file = "predict_file_drug" + str(id) + ".txt"
                patient_allsample = "patient_allsamples" + str(id) + ".txt"
                f1 = open(notmatch_patient, "w")
                f2 = open(notmatch_gene, "w")
                f3 = open(predict_file, "w")
                f4 = open(patient_allsample, 'w')

                data_test = []  # 构建了新的测试数据集，待会会把所有的挑选出来的数据都放在这里

                ###先将上述六种药物按照rf算出来的importance排序以后的特征用列表存起来###
                print("now constructing feature index...")
                sums = 0
                if feature_dir==None:
                    filefeature_prefix = "/data1/data/zengyanru/feature_oridinary_and_list/"  #TODO:其实这个也需要在外部改变
                else:
                    filefeature_prefix=feature_dir
                filefeature = "drugID" + str(id) + "_features.txt"
                filefeature = filefeature_prefix + filefeature
                f_features = open(filefeature, "r")
                featurelist = eval(f_features.read())
                featureindex = []
                for i in featurelist:
                    if sums < save_info:
                        sums += i[0]
                        featureindex.append(i[1])
                    else:
                        pass
                featureindex_sorted = sorted(featureindex)  # featureindex是rf挑选出来的这种药物的重要性比较靠前的特征索引
                real_index = eval(open("/data1/data/zengyanru/feature_oridinary_and_list/idxlist_by_sd.txt","r").read())
                real_idx_list = [real_index[i] for i in featureindex_sorted]
                featureindex_sorted = sorted(real_idx_list) #以上三行是后面加入的，为了迎合两次特征挑选！
                featurenamelist = [genenamelist[idx] for idx in
                                   featureindex_sorted]  # featurenamelist存储着基因名字，TODO：这里就是关于某一种药物的所有基因的名字了，需要到dict里面找
                #接下来这一句是将GAPDH 加到featurenamelist里面的，为了以后的标准化
                featurenamelist.append('GAPDH')
                print("features index completed.") #TODO:到时候学习，不能够加上这个特征！
                print(len(featurenamelist)-1)

                ###然后在同一药物里面找不同病人的样本信息###
                if patient_followup_name==None:
                    patient_followup_name = "/data1/data/zengyanru/findout_followups/"+str(id) + "_followup.txt" #TODO：需要外部引入
                else:
                    patient_followup_name = patient_followup_name #一定要是用自己的找xml的函数找的那种格式才行呢！
                f_followup = open(patient_followup_name, 'r')

                order = -1  # 用于记录行数，用来标记可能不匹配的样本
                print("now mapping patient's info to TCGA database...")
                overlap = []
                if cancerexp_tcga==None:
                    cancerroute = '/data1/data/TCGA/RNA_Seq_FPKM_2017_4_1/' #这里已经是左老师25服务器上的数据。认为不需要外部引入了。TODO：别人使用的话就外部引入一下吧。
                else:
                    cancerroute=cancerexp_tcga #需要是一个文件夹
                print('now establishing dict of id%d...' % id)
                nrow = 0
                for line in f_followup:
                    nrow += 1  # 记录缺失值，为了以后能够将缺失值在标准化后设置为零
                    decide = []
                    geneexplist = []
                    order += 1
                    match_count = 0
                    if line.startswith("/"):
                        cancertype = line.split('/')[5]
                        patient_name = line.split(".")[2]  # TODO：在后面需要找到病人的信息，通过索引
                        if cancertype not in overlap:
                            overlap.append(cancertype)
                            cancername = cancerroute + cancertype + '1.txt'
                            f = open(cancername, 'r')
                            cancerexpdict[cancertype] = {} #完全通过follow up信息来检索基因表达文件所在
                            for line in f:
                                line = line.split('\t')
                                symbol = line[1]  # 第一行的 symbol 就是 symbol
                                if symbol == '#N/A':
                                    pass
                                else:
                                    namesindex = line[3:]
                                    cancerexpdict[cancertype][symbol] = namesindex
                                    # print('dict of %s has done.'%cancertype)
                        else:
                            pass
                            # cancerexpdict类似于：cancerexpdict：某个癌症的名称：某个基因的名称：表达量。symbol自己存的是病人的名字

                        # print(patient_name) 只是为了测试
                        for j, k in enumerate(cancerexpdict[cancertype][
                                                  'symbol']):  # TODO:从这里开始match，每一个matching都需要试图判断，如果有没有match的，全部表达量视为0，并且输出标记，防止错误。
                            match = re.match(patient_name, k, re.I)  # 这里是match有没有这个病人的
                            if match:  # 那么 j 就是索引
                                patient_list_allsample.append(patient_name)
                                match_count += 1
                                if match_count == 1:
                                    patient_list.append(patient_name)
                                    geneexplist = []
                                    decide.append("yes")  # 判断这个样本是否在TCGA数据里面
                                    for m, l in enumerate(featurenamelist):
                                        if l in cancerexpdict[cancertype]:
                                            # print(type(cancerexpdict[cancertype][l][j]))
                                            # print(cancertype,l)
                                            # print(cancerexpdict[cancertype][l][j])#监测字典
                                            if cancerexpdict[cancertype][l][j] != '':
                                                geneexplist.append(float(cancerexpdict[cancertype][l][j].strip(
                                                    '\n')))  # 这里是每一个样本的基因表达，在新的样本中geneexplist会被替换，TODO：同样需要判断
                                            else:
                                                missing_value.append(
                                                    (nrow, m))  # m 是featurelist中的索引，也是构造出来的data_test中的索引
                                                # print('no data in dict:%s gene:%s, index is %d' % (cancertype,l,j))
                                                geneexplist.append(float(0))

                                        else:
                                            if m not in notmatch_genelist:
                                                notmatch_genelist.append(m)
                                            geneexplist.append(float(0))  # 在这里的定义，零可能是缺失值，也可能是实际值。geneexplist：是按照featurenamelist的顺序来构建的单个病人的基因表达量，针对每一种药物的。
                                    #接下来进行这个数据的标准化！！！前面featurenamelist加入了最后一个维度的GAPDH，现在要先log转换，然后在减去这个feature
                                    for index,value in enumerate(geneexplist):
                                        if value==0:
                                            geneexplist[index]=1
                                    geneexplist_log=np.log(np.array(geneexplist))
                                    denominator=geneexplist_log[-1]
                                    if denominator==0:
                                        denominator=1
                                    else:
                                        pass
                                    geneexplist_more_one_feature=geneexplist_log/denominator
                                    geneexplist=geneexplist_more_one_feature[:len(geneexplist_more_one_feature)-1]

                                    data_test.append(geneexplist)

                                    # print(geneexplist)  ##TODO:need to del later!!!!!!!!!!!!!!!!!!!!!!!!!!

                        if "yes" in decide:  # 为了查看某个病人的数据是否在数据库中，不在的话就用 0 代替
                            pass
                        else:
                            patient_list.append(patient_name)
                            # print(order)
                            print(order,
                                  file=f1)  # TODO:稍后需要将这个order输出到文件中以标记没有匹配的样本，order代表了从表头开始（第 0 行，表头的order=0，第一个样本的order=1）,f1
                            geneexplist = [0 for nbs in range(len(featurenamelist)-1)]  # TODO：原本是1，现在改成0
                            data_test.append(geneexplist)
                            # print(len(geneexplist))
                print(notmatch_genelist, file=f2)  # TODO:需要一个新的文件来讲述哪个基因没有匹配上,f2

                print('data for drug %s has completed.' % id)
                del cancerexpdict
                print("data is ready,now begin machine learning and predict...")
                data_test = np.array(data_test)
            return data_test
        elif from_makeup_data == True and isMicroarray == True and for_pick_feature_test==False:
            if microarray_dir==None:
                microarray_dir="/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/"
            else:
                microarray_dir=microarray_dir

            if feature_dir == None:
                name_prefix = '/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/feature_oridinary_and_list/'
            else:
                name_prefix = feature_dir

            # drug_pick=self.drug_pick
            drug_pick=drug_id

            for i in drug_pick:
                print('now constructing feature index list...')
                name_suffix = 'drugID' + str(i) + '_features.txt'
                names = name_prefix + name_suffix
                f_features = open(names, 'r')  # close later in order to release memory
                features_list = eval(f_features.read())
                sum = 0
                featureID_list = []  # del later in order to refresh

            for j in features_list:
                if sum < save_info:
                    sum += j[0]
                    featureID_list.append(j[1])  # 这里的格式是按照importance大小来排序的关于特征的index，存在列表里面
                else:
                    pass

            featureID_list_sorted = sorted(featureID_list)
            # print(featureID_list_sorted)  #TODO:看sorted是不是按照数字来排序的！
            print('feature list has been finished')
            # 在此处准备读取进行机器学习的数据，并且用randomforest来做，保存214个模型
            print('now begin to load drug data...')
            datatarget_prefix = microarray_dir
            dataname_suffix = 'drug' + str(i) + 'data.txt'
            targetname_suffix = 'drug' + str(i) + 'target.txt'
            dataname = datatarget_prefix + dataname_suffix
            targetname = datatarget_prefix + targetname_suffix
            f_data = open(dataname, 'r')  # close later in order to release memory
            f_target = open(targetname, 'r')  # close later in order to release memory
            data_list = eval(f_data.read())  # del later
            target_list = eval(f_target.read())  # del later
            # 接下来对于data进行标准化一下，因为以往构造的data的模式比较特别，GAPDH已经找到在索引3756的位置，可以直接使用
            print('now begin to normalize the data...')
            gapdh_exp = []

            for i in data_list:
                gapdh_exp.append(i[3756])

            gapdh_rev = 1 / np.array(gapdh_exp)
            data_list = np.array(data_list)

            for i, j in enumerate(data_list):
                data_list[i] = data_list[i] * gapdh_rev[i]

            data = []
            target = [float(l) for l in target_list]
            print('now compressing data dims...')

            for k in data_list:
                data_oneelement = [k[idx] for idx in featureID_list_sorted]
                data.append(data_oneelement)

            data = np.array(data)
            target = np.array(target)
            eight_two = len(data) // data_split
            train = len(data) - eight_two
            data_train = data[:train]
            target_train = target[:train]
            data_test = data[train:]
            target_test = target[train:]
            inputdim = len(data[0])
            y_train = np.array(target_train).reshape(-1, 1)
            y_test = np.array(target_test).reshape(-1, 1)
            print('data has been loaded and begin to build a model...')
            return data_train,y_train,data_test,y_test
        elif for_pick_feature_test==True:
            if microarray_dir==None:
                microarray_dir="/data/zengyanru/about_drug_sensitive/drug_cellmorethan800"
            else:
                microarray_dir=microarray_dir
            name_prefix = '/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/feature_oridinary_and_list/'

            # drug_pick=self.drug_pick
            drug_pick=drug_id

            for i in drug_pick:
                print('now constructing feature index list...')
                name_suffix = 'drugID' + str(i) + '_features.txt'
                names = name_prefix + name_suffix
                f_features = open(names, 'r')  # close later in order to release memory
                features_list = eval(f_features.read())
                sum = 0
                featureID_list = []  # del later in order to refresh

            for j in features_list:
                if sum < save_info:
                    sum += j[0]
                    featureID_list.append(j[1])  # 这里的格式是按照importance大小来排序的关于特征的index，存在列表里面
                else:
                    pass

            featureID_list_sorted = sorted(featureID_list)
            # print(featureID_list_sorted)  #TODO:看sorted是不是按照数字来排序的！
            print('feature list has been finished')
            # 在此处准备读取进行机器学习的数据，并且用randomforest来做，保存214个模型
            print('now begin to load drug data...')

            data_list = test_data
            target_list = test_target
            # 接下来对于data进行标准化一下，因为以往构造的data的模式比较特别，GAPDH已经找到在索引3756的位置，可以直接使用
            print('now begin to normalize the data...')
            gapdh_exp = []

            for i in data_list:
                gapdh_exp.append(i[3756])

            gapdh_rev = 1 / np.array(gapdh_exp)
            data_list = np.array(data_list)

            for i, j in enumerate(data_list):
                data_list[i] = data_list[i] * gapdh_rev[i]

            data = []
            target = [float(l) for l in target_list]
            print('now compressing data dims...')

            for k in data_list:
                data_oneelement = [k[idx] for idx in featureID_list_sorted]
                data.append(data_oneelement)

            data = np.array(data)
            target = np.array(target)
            eight_two = len(data) // 5
            train = len(data) - eight_two
            data_train = data[:train]
            target_train = target[:train]
            data_test = data[train:]
            target_test = target[train:]
            inputdim = len(data[0])
            y_train = np.array(target_train).reshape(-1, 1)
            y_test = np.array(target_test).reshape(-1, 1)
            print('data has been loaded and begin to build a model...')
            del data_list
            del target_list
            return data_train,y_train,data_test,y_test

    def zyr_mlp(self,data_train,target_train,data_test,target_test,learning_rate=0.01,epochs=150,batch_size=15,isSave=False,save_in_path='',drug_id=None,optimizer='adam'):
        #optimizer必须是adam 或者 sgd 二选一
        #kernel_regularizer必须是类似kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
        #save_in_path必须是文件夹，注意加上/
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD
        from keras.optimizers import Adam
        from keras.callbacks import ModelCheckpoint
        from scipy import stats
        from keras import regularizers

        inputdim=len(data_train[0])

        model = Sequential()
        model.add(Dense(inputdim // 2, input_dim=inputdim, init='uniform',
                        kernel_regularizer=regularizers.l2(0.0001)))  # 输入层，28*28=784，最好附加初始化，用identity
        model.add(Activation('relu'))  # 激活函数是tanh(后面变成了relu因为对mnist的处理结果会好一些)
        model.add(
            Dense(inputdim // 4, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l1(0.0001)))
        model.add(Activation('relu'))
        model.add(
            Dense(inputdim // 6, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l1(0.0001)))
        model.add(Activation('relu'))
        model.add(Dense(1))  # 输出结果和药物对应的话只是一维的
        model.add(Activation('linear'))  # 最后一层linear，因为是实际的结果

        if optimizer=='adam':
            adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='mse', optimizer=adam, metrics=["mse"])
        elif optimizer=='sgd':
            sgd = SGD(lr=learning_rate)
            model.compile(loss='mse', optimizer=sgd, metrics=["mse"])

        if isSave:
            modelname=save_in_path+'model_mlp_fordrug'+str(drug_id)+'.h5'
            check_test = ModelCheckpoint(modelname, monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
            model.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,callbacks=[check_test])
            predicted = model.predict(data_test)
            predicted = predicted.reshape(-1)
            pearson = stats.pearsonr(predicted, target_test.reshape(-1))

        else:
            model.fit(data_train, target_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
            predicted = model.predict(data_test)
            predicted = predicted.reshape(-1)
            pearson = stats.pearsonr(predicted, target_test.reshape(-1))

        del model
        return pearson









