'''from KGML_parser_zyr import *
pathway=read(open('D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa00240_RRM1.xml','rU'))
# for r in list(pathway.entries.items())[:2]:
#     print(r)
# for r in pathway.genes[:2]:
#     print(r.graphics[0].name)
for r in pathway.relations[:1]:
    print(r.subtypes[0][0])
    print(r._entry1)
    '''
import numpy as np
from yanru.zyr_KGML_parser.KGML_parser_zyr import *

# gene_name_list=eval(open('D:\zengyr\drugsensitivity_rawdata\geneexpression_namelist.txt','r').read()) #传入基因名字顺序的列表，为了查找相应的基因表达量。

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
        # print(relation_dict[str(r._entry1)])


    # print(relation_dict)
    return relation_dict

#构建reaction的dict，这个函数只用在酶的pathway里面
# 格式如同：基因id：reversible（或者irreversible）
def construct_reaction_dict(KGML_file):
    assert type(KGML_file) == str
    pathway = read(open(KGML_file, 'rU'))
    reaction_dict={}
    for r in pathway.reactions:
        reaction_dict[r._id]=r.type
    # print(reaction_dict)
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
                                # gene_expression2=gene_dict[genename2][1]
                                #这里是正式修改的步骤
                                init_matrix[indx_genename1,indx_genename2]=gene_expression1
                                # print(gene_expression1)
                    else:
                        pass
    # print(init_matrix)
    return init_matrix



if __name__ == '__main__':

    pathway='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa00240_RRM1.xml'
    pathway1='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\hsa01521_EGFR.xml'
    filepath='D:\zengyr\\about_drug_sensitivity\kegg_pathway_map_somedrugs\zyr_KGML_parser\\test_kgml.txt'
    # filepath='D:\zengyr\\about_drug_sensitivity\Cell_line_RMA_proc_basalExp_clear.txt'
    data,indx=construct_data_matrix(filepath)
    # print(indx)
    # data_matrix,geneindex=construct_data_matrix(filepath)
    # construct_geneid_dict(pathway,data_list_per_patient=data_matrix[0],genename_index=geneindex)

    genedict=construct_geneid_dict(pathway,data[0],genename_index=indx)
    geneiddict=construct_id_name_map(pathway)
    relation_dict=construct_relation_dict(pathway)
    reaction_dict=construct_reaction_dict(pathway)
    # print(construct_matrix_for_enzymes_RWR(genedict,geneid_dict=geneiddict,relation_dict=relation_dict,reaction_dict=reaction_dict))
    # print('yes'if i[0]!=0 for i in construct_matrix_for_gene_RWR(genedict,geneid_dict=geneiddict,relation_dict=relation_dict))
    count=0
    # num=len(construct_matrix_for_gene_RWR(genedict,geneid_dict=geneiddict,relation_dict=relation_dict)[0])
    # summa=np.sum(construct_matrix_for_gene_RWR(genedict,geneid_dict=geneiddict,relation_dict=relation_dict),axis=0)
    summa=np.sum(construct_matrix_for_enzymes_RWR(genedict,geneid_dict=geneiddict,relation_dict=relation_dict,reaction_dict=reaction_dict),axis=0)
    print(summa)
    '''for i in summa:
        if i!=0:
            count+=1
    print(count,num)'''