# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:14
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : RWR_for_pathway.py
# @Software: PyCharm

import numpy as np
from yanru.zyr_KGML_parser.KGML_parser_zyr import *

def construct_data_matrix(filepath,skiprows=1,skipcols=2,delimiter='\t',unpack=True,genename_in_col=1):
    #读取矩阵的函数，比np.loadtxt更符合我的要求，参数都顾名思义。额外添加了记录基因index的几行语句
    assert type(filepath)==str
    f_data=open(filepath,'r')
    patient_nums_and_tags=len(f_data.readline().split('\t'))
    patient_nums=[i+skipcols for i in range(patient_nums_and_tags-skipcols)]
    data=np.loadtxt(filepath,delimiter=delimiter,skiprows=skiprows,usecols=patient_nums,unpack=unpack) #unpack是看转置与否的
    #下面是建立每个基因表达矩阵的基因索引
    genename_index=[]
    #这里需要注意的是，由于上方有了readline()，下方其实直接从第二行开始索引的。
    for line in f_data:
        split_line=line.split(delimiter)
        genename_index.append(split_line[genename_in_col-1])
    return (data,genename_index)

# 创建一个entry id dict，将KGML文件中的基因和各自的表达量存储在entry dict中，如果没有这样的表达量的话，就取值为1
#格式如同：基因：（基因id，基因表达量）
def construct_geneid_dict(KGML_file,data_list_per_patient=None,genename_index=None):
    #KGML_file是KGML文件所在路径；data_matrix是参考的基因表达矩阵;genename_index是一个按照顺序存储的基因名字的list
    assert type(KGML_file)==str
    pathway=read(open(KGML_file,'rU'))
    #找基因
    gene_dict={}

    for r in pathway.genes:
        value_id=r._id
        keys_string=r.graphics[0].name
        keys=keys_string.split(', ')
        for i,j in enumerate(keys):
            if '...' in j:
                keys[i]=keys[i].strip('...')
        #接下来从gene matrix里面找相关基因的表达水平，在下方重新定义一个函数吧
        for k in keys:
            value_exp=find_corresp_geneexp(k,data_list_per_patient,genename_index)
            gene_dict[k]=(value_id,value_exp)

    # print(gene_dict)
    return gene_dict

#为了索引方便，还搞一个存储了id的，id对应基因名字的函数吧。
#格式如同：基因id：基因名字列表
def construct_id_name_map(KGML_file):
    assert type(KGML_file) == str
    pathway = read(open(KGML_file, 'rU'))
    geneid_dict={}
    for r in pathway.genes:
        value_id=r._id
        keys_string = r.graphics[0].name
        keys = keys_string.split(', ')
        for i, j in enumerate(keys):
            if '...' in j:
                keys[i] = keys[i].strip('...')
        geneid_dict[value_id]=keys
    return geneid_dict

# 创建relation的dict，用于以后查找相关关系，并且写入新的矩阵
#格式如同：基因id：[（反应的基因id，激活1（不激活0））,(...)]
def construct_relation_dict(KGML_file):
    assert type(KGML_file) == str
    pathway = read(open(KGML_file, 'rU'))
    relation_dict={}
    #下面要筛选正确的关系，如果是像mapLink, protein-conpound那种就算了。relation中需要将这些去除。
    for r in pathway.relations:
        if r.type!='maplink':
            relationlist=[c[0] for c in r.subtypes]
            # 将一对多的关系也记录下来
            if 'inhibition' in relationlist:
                if r._entry1 in relation_dict:
                    relation_dict[r._entry1].append((r._entry2, -1))
                else:
                    relation_dict[r._entry1]=[(r._entry2,-1)]
            else:
                if r._entry1 in relation_dict:
                    relation_dict[r._entry1].append((r._entry2, 1))
                else:
                    relation_dict[r._entry1]=[(r._entry2,1)]
    return relation_dict

#构建reaction的dict，这个函数只用在酶的pathway里面
# 格式如同：基因id：reversible（或者irreversible）
def construct_reaction_dict(KGML_file):
    assert type(KGML_file) == str
    pathway = read(open(KGML_file, 'rU'))
    reaction_dict={}
    for r in pathway.reactions:
        reaction_dict[r._id]=r.type
    return reaction_dict


# 找到某个基因的表达量
def find_corresp_geneexp(gene,data_list_per_patient,genename_index):
    assert type(gene)==str
    if gene in genename_index:
        idx=genename_index.index(gene)
        expression=data_list_per_patient[idx]
    else:
        expression=1
    return expression

#这里准备构建matrix了
#为了一步到位，其他函数也在这里调用比较好
#为了以后能重复实验，将基因按照字母来排序比较好
def construct_matrix_for_enzymes_RWR(gene_dict,geneid_dict,relation_dict=None,reaction_dict=None,save_genename_list_path=None):
    #这个函数专门给酶的map的，注意，不是gene的
    #这里接受的gene_dict中的表达量数据，只代表一个病人的！！注意后来输入的格式！
    genelist=sorted([i for i in gene_dict]) #到时候可以在这里找index来修改矩阵
    if save_genename_list_path==None:
        pass
    else:
        fw_save_genename_list=open(save_genename_list_path,'w')
        print(genelist,file=fw_save_genename_list)
    init_matrix=np.zeros(len(genelist)*len(genelist)).reshape(len(genelist),len(genelist))
    #下面进入正式构造环节
    for id in relation_dict:
        upstreamid=id
        if upstreamid in geneid_dict:
            upstream_gene=geneid_dict[upstreamid]
            for genename1 in upstream_gene:
                indx_genename1=genelist.index(genename1)
                gene_expression1=gene_dict[genename1][1]
                for downstream_id in relation_dict[id]:
                    downstreamid = downstream_id[0]
                    activation=downstream_id[1]
                    if activation==1:
                        if downstreamid in geneid_dict:
                            gene_this_relation=geneid_dict[downstreamid]
                            for genename2 in gene_this_relation:
                                indx_genename2=genelist.index(genename2)
                                gene_expression2=gene_dict[genename2][1]
                                #这里是正式修改的步骤
                                init_matrix[indx_genename1,indx_genename2]=gene_expression1
                                # print(gene_expression2)
                                # 现在验证这个reaction是否为reversible的,如果是reversible的话，则index转置的部分使用genename2的表达来表示链接权重
                                if reaction_dict[upstreamid] == 'reversible' and reaction_dict[downstreamid] == 'reversible':
                                    init_matrix[indx_genename2, indx_genename1] = gene_expression2
                    else:
                        pass
    return init_matrix

#接下来构造关于gene的map的矩阵。少考虑一些因素，那个因素就是reaction，不用考虑是否可逆。
#注意，图中的一些compound是没有考虑在内的，因为没有相应的compound 的表达量，因此直接从基因跳到基因
def construct_matrix_for_gene_RWR(gene_dict,geneid_dict,relation_dict=None,save_genename_list_path=None):
    #这个函数专门给酶的map的，注意，不是gene的
    #这里接受的gene_dict中的表达量数据，只代表一个病人的！！注意后来输入的格式！
    genelist=sorted([i for i in gene_dict]) #到时候可以在这里找index来修改矩阵
    if save_genename_list_path==None:
        pass
    else:
        fw_save_genename_list=open(save_genename_list_path,'w')
        print(genelist,file=fw_save_genename_list)
    init_matrix=np.zeros(len(genelist)*len(genelist)).reshape(len(genelist),len(genelist))
    #下面进入正式构造环节
    for id in relation_dict:
        upstreamid=id
        if upstreamid in geneid_dict:
            upstream_gene=geneid_dict[upstreamid]
            for genename1 in upstream_gene:
                indx_genename1=genelist.index(genename1)
                gene_expression1=gene_dict[genename1][1]
                for downstream_id in relation_dict[id]:
                    downstreamid = downstream_id[0]
                    activation=downstream_id[1]
                    if activation==1:
                        if downstreamid in geneid_dict:
                            gene_this_relation=geneid_dict[downstreamid]
                            for genename2 in gene_this_relation:
                                indx_genename2=genelist.index(genename2)
                                init_matrix[indx_genename1,indx_genename2]=gene_expression1

                    else:
                        pass
    # print(init_matrix)
    return init_matrix

if __name__ == '__main__':

    pathway='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa00240_RRM1.xml'
    pathway1='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa01521_EGFR.xml'
    filepath='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\zyr_KGML_parser\\test_kgml.txt'
    data,indx=construct_data_matrix(filepath)
    genedict=construct_geneid_dict(pathway,data[0],genename_index=indx)
    geneiddict=construct_id_name_map(pathway)
    relation_dict=construct_relation_dict(pathway)
    reaction_dict=construct_reaction_dict(pathway)
    count=0
    summa=np.sum(construct_matrix_for_enzymes_RWR(genedict,geneid_dict=geneiddict,relation_dict=relation_dict,reaction_dict=reaction_dict),axis=0)
    print(summa)

####下面是算法实现##############
import numpy as np
from yanru.zyr_KGML_parser.construct_matrix import construct_reaction_dict
from yanru.zyr_KGML_parser.construct_matrix import construct_relation_dict
from yanru.zyr_KGML_parser.construct_matrix import construct_id_name_map
from yanru.zyr_KGML_parser.construct_matrix import construct_data_matrix
from yanru.zyr_KGML_parser.construct_matrix import construct_geneid_dict
from yanru.zyr_KGML_parser.construct_matrix import find_corresp_geneexp
from yanru.zyr_KGML_parser.construct_matrix import construct_matrix_for_enzymes_RWR
from yanru.zyr_KGML_parser.construct_matrix import construct_matrix_for_gene_RWR
import tensorflow as tf
from yanru.dataIterator import DataIterator
from sklearn import linear_model

def RWR(matrix,restart_prob=0.5,max_iter=1000,precision=0.0000000001): # 文献中的precision也是用1e-10.彭凯师兄给的文献
    sum_array=np.sum(matrix,axis=0)
    sum_list=[]
    for sums in sum_array:
        if sums==0:
            sum_list.append(1)
        else:
            sum_list.append(sums)
    sum_array=np.matrix(sum_list)

    #将matrix中的元素归一化到0-1之间，表示转移的概率
    matrix_prob=matrix/sum_array

    #这里先定义更新的函数
    def cal_a_step(st,s0):
        assert len(st)==len(s0)
        st_next=(1-restart_prob)*np.dot(st,matrix_prob)+restart_prob*s0
        return st_next

    #注意初始化matrix
    init_matrix=np.zeros([len(matrix[0]),len(matrix[0])])
    #这里需要注意break。以下是真正的RWR算法。
    for gene_idx, s in enumerate(matrix):
        s0=s
        st=s
        for i in range(max_iter):
            # print(i)
            st=np.array(cal_a_step(st,s0))[0]
            L1_norm_list=cal_a_step(st,s0)-st
            L1_norm=sum(abs(i) for i in np.array(L1_norm_list)[0])
            if L1_norm<=precision:
                # print('yes')
                #精度足够了就写入矩阵
                init_matrix[gene_idx]=cal_a_step(st,s0)
                break #也就是到达了精度，或者到达了循环次数就会推出循环。和平时的控制是一样的。
        #循环足够了就写入矩阵
        init_matrix[gene_idx]=cal_a_step(st,s0)

    return init_matrix

def context_node_feature_SVDversion(matrix_for_approaching,first_d=10):
    length=len(matrix_for_approaching[0])
    Q_matrix=np.array([1/length]).repeat(length*length).reshape(length,length)
    L_matrix=np.log((matrix_for_approaching+Q_matrix))-np.log(Q_matrix)

    from numpy import linalg
    u,s,v=linalg.svd(L_matrix)

    u_pickout_d_vector_transpose=u[:first_d]
    u_pickout_d_vector=np.transpose(u_pickout_d_vector_transpose) #这里的操作是按照SVD自己的得出的U,S,V 的转置与否来处理的

    s_first_d_value=s[:first_d]

    v_pickout_d_vector_transpose=v[:first_d]
    v_pickout_d_vector=np.transpose(v_pickout_d_vector_transpose)

    X=np.dot(u_pickout_d_vector,np.sqrt(s_first_d_value))
    W=np.dot(v_pickout_d_vector,np.sqrt(s_first_d_value))

    return X,W

def context_node_feature_KLversion(matrix_for_approaching,learning_rate=0.01,sample_num=1000,batch_size=15,epochs=100,display_step=1):
    length_of_w_and_x=len(matrix_for_approaching[0])
    #创建一个假的数据集，保存所有零矩阵的x，和所有元素都是matrix_for_approaching的y
    data=[]
    target=[]
    for i in range(sample_num):
        data.append(np.array(np.zeros([length_of_w_and_x,length_of_w_and_x]).reshape(-1)))
        target.append(np.array(matrix_for_approaching.reshape(-1)))
    data=np.array(data)
    target=np.array(target)
    # print(data)
    #创建W 和X ，为后续计算做准备。按照文献，从【-0.05,0.05】的uniform中抽取
    x=tf.placeholder('float',[None,length_of_w_and_x*length_of_w_and_x]) #假装有一个placeholder，用零矩阵代替即可，用相加的模式。也可以单位矩阵然后相乘什么的。
    y=tf.placeholder('float',[None,length_of_w_and_x*length_of_w_and_x])

    W=tf.Variable(tf.random_uniform([1,length_of_w_and_x],minval=-0.05,maxval=0.05),name='W')
    X=tf.Variable(tf.random_uniform([1,length_of_w_and_x],minval=-0.05,maxval=0.05),name='X')

    pred=tf.add(tf.matmul(W,X,transpose_a=True),x) #预测的函数
    KL_divergency=tf.reduce_sum(pred*tf.log(pred/y)) #这个就是cost，用于往后测试

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(KL_divergency)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(W))
    for eps in range(epochs):
        train_data = DataIterator(matrix_data=data, label_data=target, batchSize=batch_size)
        avg_cost = 0
        while train_data.isHasNext:
            batch_x, batch_y = train_data.next_batch()
            # sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
            ops,c = sess.run([optimizer,KL_divergency], feed_dict={x: batch_x, y: batch_y})
            # c = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            # out_test = sess.run(pred, feed_dict={x:batch_x})
            # print(c)
            avg_cost += c
        if eps % display_step == 0:
            avg_cost = avg_cost / (len(data) // batch_size)
            # weight=sess.run(W)
            print('Epoch:', '%04d' % (eps + 1), 'cost=', '{:.9f}'.format(avg_cost))
    # print(sess.run(W))
if __name__ == '__main__':
    #测试一下
    pathway='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa00240_RRM1.xml'
    pathway1='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa01521_EGFR.xml'
    filepath='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\zyr_KGML_parser\\test_kgml.txt'
    data,indx=construct_data_matrix(filepath)

    genedict=construct_geneid_dict(pathway,data[0],genename_index=indx)

    geneiddict=construct_id_name_map(pathway)

    relation_dict=construct_relation_dict(pathway)

    reaction_dict=construct_reaction_dict(pathway)

    enzyme_matrix=construct_matrix_for_enzymes_RWR(genedict,geneiddict,relation_dict,reaction_dict)

    finalmatrix=RWR(matrix=enzyme_matrix)

    # context_node_feature_KLversion(finalmatrix)
    print(context_node_feature_SVDversion(finalmatrix))

