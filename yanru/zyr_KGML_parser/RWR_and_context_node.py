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

