import numpy as np
import tensorflow as tf
import os


def save_info_list(filename, list):
    fid = open(filename, 'w')
    for elem in list:
        # fid.write(str(elem)+'\n')
        fid.write(str(elem)+'\n')
    fid.close()


def createDir(Dir):
    if os.path.exists(Dir) is False:
        os.mkdir(Dir)


def load_nets_params(modelname, in_size, out_size):
    g = tf.Graph()
    weights_params, bias_params = None, None
    with g.as_default():
        weights = tf.Variable(tf_xavier_init(in_size, out_size, const=1)
                              , dtype=tf.float32, name='weights')
        bias = tf.Variable(tf.zeros([out_size]), dtype=tf.float32, name='hidden_bias')
        sess = tf.Session()
        new_saver = tf.train.Saver({'weights': weights, 'hidden_bias': bias})
        new_saver.restore(sess, modelname)
        weights_params, bias_params = sess.run('weights:0'), sess.run('hidden_bias:0')

    params = {'weights': weights_params, 'bias': bias_params}
    return params


def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)


def combine_dicts(dict_list):
    new_dict = {}
    dict_num = len(dict_list)
    for key in dict_list[0].keys():
        new_dict[key] = []
        for i in range(dict_num):
            new_dict[key].append(dict_list[i][key])
    return new_dict


def windows2linuxDirs(winDirs, isLinux):
    newDir = ''
    if isLinux is True:
        newDir = winDirs.replace('\\', '/')
    else:
        newDir = winDirs
    return newDir


def writeList2txt(PATH, list):
    fwriter = open(PATH, 'w')
    for i in range(len(list)):
        if i != len(list)-1:
            fwriter.write(str(list[i])+'\n')
        else:
            fwriter.write(str(list[i]))
    fwriter.close()


def writeResult2txt(PATH, columnData):
    fwriter = open(PATH, 'w')
    for i in range(columnData.shape[0]):
        if i != columnData.shape[0]-1:
            fwriter.write(str(columnData[i, 0])+'\n')
        else:
            fwriter.write(str(columnData[i, 0]))
    fwriter.close()


def dataDict2Matrix(data_dict, samples, genes):
    matrix = np.zeros([len(samples), len(genes)])
    for row in range(len(samples)):
        for col in range(len(genes)):
            matrix[row][col] = data_dict[samples[row]][genes[col]]
    return matrix


def save_matrix_file(PATH, data_dict, gene_list, sample_1):
    fwrite = open(PATH, 'w')
    fwrite.write('sample_name')
    for tmp_gene in gene_list:
        fwrite.write('  '+tmp_gene)
    fwrite.write('\n')

    for tmp_sample in sample_1:
        fwrite.write(tmp_sample)
        for tmp_gene in gene_list:
            tmp_value = data_dict[tmp_sample][tmp_gene]
            fwrite.write('  '+tmp_value)
        fwrite.write('\n')
    fwrite.close()


def save_pure_matrix(PATH, matrix):
    fwrite = open(PATH, 'w')
    [sample_num, dim] = np.shape(matrix)
    for row in range(sample_num):
        tmp_line = str(matrix[row][0])
        for col in range(1, dim):
            tmp_line = tmp_line+' '+str(matrix[row][col])
        fwrite.write(tmp_line+'\n')
    fwrite.close()


def save_label_file(PATH, label_dict, sample):
    fwrite = open(PATH, 'w')
    fwrite.write('sample_name    label\n')
    for tmp_sample in sample:
        fwrite.write(tmp_sample+'  '+str(label_dict[tmp_sample])+'\n')
    fwrite.close()


def read_matrix_data(filename, sample_num):
    fread = open(filename)
    gene_line = fread.readline()
    gene_dim = len(gene_line.split())-1
    import numpy as np
    data_matrix = np.zeros([sample_num, gene_dim])
    line = fread.readline()
    count = 0
    while line:
        tmp_arr = line.split()
        num_arr = []
        for a in range(1, len(tmp_arr)):
            num_arr.append(float(tmp_arr[a]))
        data_matrix[count, :] = num_arr
        count += 1
        line = fread.readline()
    Size = np.shape(data_matrix)
    print('matrix height:'+str(Size[0])+', width:'+str(Size[1]))
    return data_matrix


def save_pca_matrix(PATH, pca_matrix):
    [sample_num, feat_num] = np.shape(pca_matrix)
    fwrite = open(PATH, 'w')
    for row in range(sample_num):
        for col in range(feat_num-1):
            fwrite.write(str(pca_matrix[row][col])+' ')
        fwrite.write(str(pca_matrix[row][col])+'\n')
    fwrite.close()


def load_np_matrix(PATH, sample_num, feat_dim):
    fread = open(PATH)

    data_matrix = np.zeros([sample_num, feat_dim])
    line = fread.readline()
    count = 0
    while line:

        tmp_arr = line.split()
        num_arr = []
        for a in range(len(tmp_arr)):
            num_arr.append(float(tmp_arr[a]))
        data_matrix[count, :] = num_arr
        count += 1
        line = fread.readline()
    Size = np.shape(data_matrix)
    print('matrix height:' + str(Size[0]) + ', width:' + str(Size[1]))
    return data_matrix


def load_np_label(PATH, sample_num):
    fread = open(PATH)
    label_matrix = np.zeros([sample_num, 1])
    count = 0
    line = fread.readline()
    while line:
        label_matrix[count, 0] = float(line.replace('\n', ''))
        count += 1
        line = fread.readline()
    return label_matrix


def load_list(filename):
    fid = open(filename, 'r')
    List = []
    curline = fid.readline()
    while curline:
        List.append(float(curline.replace('\n', '')))
        curline = fid.readline()
    fid.close()
    return List


