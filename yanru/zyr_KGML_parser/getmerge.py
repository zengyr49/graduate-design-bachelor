import numpy as np
from yanru.zyr_KGML_parser.process_of_analyzing_drug_sensitivity import analyzing_drug_sensitivity
analyzing_drug_sensitivity=analyzing_drug_sensitivity()
def get_merge(data_vector):
    indx_microarray = eval(
        open("/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/geneexpression_namelist.txt", "r").read())
    xml_path = "/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/drug_target_pathway/hsa01524_TOP2.xml"
    X, W = analyzing_drug_sensitivity.add_pathway_data(data_vector=data_vector, indx=indx_microarray, xml_path=xml_path,
                                                       pathway_type='genes', max_iter=100, precision=0.000001)
    X = X.reshape(-1)
    W = W.reshape(-1)
    merge = np.array(np.hstack((X, W)))[0]  # 主要是matrix索引真麻烦
    return merge