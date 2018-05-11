import numpy as np
import numpy
def load_and_normalized(drug_id=134,isMicroarray = True):
    #原本想把load RNAseq的也加进来的，后面发现好像意义不大，于是就舍弃了。到时候直接在process of analyzing drug sensitivity中load就行。
    # 前期主要在microarray data上做
    #使用134号药物作为测试吧！
    if isMicroarray == True:
        drug_id=drug_id
        database_path = "/data/zengyanru/about_drug_sensitive/drug_cellmorethan800/"
        data_to_load = database_path + "drug" + str(drug_id) + "data.txt"
        target_to_load = database_path + "drug" + str(drug_id) + "target.txt"
        f_data = open(data_to_load,"r")
        f_target = open(target_to_load,"r")
        data = eval(f_data.read())
        target_str = eval(f_target.read())
        target = [float(i) for i in target_str]
        del target_str
        cut_avg = sum(target)/len(target)
        target_class = [] #这个是给机器学习用的；
        # target_sk = [] #这个是给sklearn一行用的；
        for i in target:
            if i >= cut_avg:
                target_class.append([1,0]) #大于均值，说明不敏感，说明是阴性，阴性用[1,0]表示；
                # target_sk.append(0)
            else:
                target_class.append([0,1]) #小于均值，敏感，阳性，用[0,1]表示；
                # target_sk.append(1)
        gapdh_exp = []
        for i in data:
            gapdh_exp.append(i[3756])
        gapdh_rev = 1 / np.array(gapdh_exp)
        data = np.array(data)
        for i, j in enumerate(data):
            data[i] = data[i] * gapdh_rev[i]
        target_class = np.array(target_class)
        data = np.array(data)
    return data,target_class

def pretreat_by_nmf(data):
    #输入的格式必须是numpy的data，也就是数组，可以使matrix 也可以是 array
    import sklearn.decomposition.nmf as nmf
    nmf_model = nmf.NMF(n_components=300, max_iter=5000, tol=0.05)
    data_out=nmf_model.fit_transform(data)
    return data_out

def pretreat_by_pca(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=300)
    data_out=pca.fit_transform(data)

    return data_out

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



def pretreat_by_pso(data_all,target_all):
    #这个函数运行超级久，久达好几天。target_all 务必是二值的那种，也就是one hot 以后的
    import numpy as np
    from yanru.pso_yanru_forbinary import pso
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD
    from keras.optimizers import Adam
    import numpy
    from sklearn.externals import joblib
    import random
    from sklearn import metrics
    import keras.backend as K
    from keras.callbacks import ReduceLROnPlateau
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    import numpy as np
    from scipy import stats

    x_all_test1 = data_all[700:]
    x_all_train1 = data_all[:700]
    y_all_train1 = target_all[:700]
    y_all_test1 = target_all[700:]

    x_all_train1 = numpy.array(x_all_train1)
    y_all_train1 = numpy.array(y_all_train1)
    x_all_test1 = numpy.array(x_all_test1)
    y_all_test1 = numpy.array(y_all_test1)

    def weight(x, *args):  # TODO:actually, x is what we want! 我们还需要将整个机器学习的框架都改造了，并且加入计算AUC的代码！
        x_all_train1, x_all_test1 = args  # TODO:remenber to transfer the data into array
        # creatvar=locals()
        # for i in range(len(x_all_test1[0])):
        # creatvar['w'+str(i)]=x[i]  #这里建立了和数据一样行数的w，为的是将每个w和数据本来的模型相乘。总体应该是900个，样本应该是90个；


        x_all_train2 = x_all_train1 * x
        x_all_test2 = x_all_test1 * x
        x_all_train2 = x_all_train2.tolist()
        x_all_test2 = x_all_test2.tolist()
        # print(x_all_train2[0][0:50])

        # print(x_all_train2[0][0:50])


        print('now eliminating zeros...')
        # 以下是将非零的维度抽提出来
        indx = []
        for i, j in enumerate(x):
            if j == 1:
                indx.append(i)
        print('index has been constructed')

        for i, j in enumerate(x_all_train2):
            # x_all_train2[i]=sorted(set(j),key=j.index)
            # print('done',i)
            x_all_train2[i] = [j[k] for k in indx]

        for i, j in enumerate(x_all_test1):
            # x_all_test2[i] = sorted(set(j), key=j.index)
            # print('done',i)
            x_all_test2[i] = [j[k] for k in indx]

        x_all_train2 = np.array(x_all_train2)
        x_all_test2 = np.array(x_all_test2)
        # print(x_all_train2[0][0:50])

        # TODO:add an algorithm that applys to our data(all_dimensions), actually mlp
        #######用作写报告哦#######
        int_dim = len(x_all_train2[0])

        model = Sequential()
        model.add(Dense(int_dim // 2, input_dim=int_dim, init='uniform'))  # 输入层，17419，最好附加初始化，用uniform
        model.add(Activation('relu'))  # 激活函数是tanh(后面变成了relu因为对mnist的处理结果会好一些)
        # model.add(Dropout(0.2))  # 采用50%的dropout

        model.add(Dense(int_dim // 10))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(int_dim // 100))
        model.add(Activation('relu'))

        model.add(Dense(int_dim // 1000))
        model.add(Activation('relu'))

        model.add(Dense(int_dim // 2000))
        model.add(Activation('relu'))


        model.add(Dense(1))  # 输出结果和药物对应的话只是二维的，结果会有one hot 编码
        model.add(Activation('linear'))  # 最后一层sigmoid，因为是实际的结果

        # 设定学习率（lr）等参数
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, verbose=0, mode='min', epsilon=0.0001,
        #                               cooldown=0, min_lr=0)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
        # check_test = ModelCheckpoint('best_model_test.h5', monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
        hist = model.fit(x_all_train2, y_all_train1, batch_size=15, epochs=100, shuffle=True, verbose=2)
        # model = load_model('best_model_test0518.h5')
        predict = model.predict(x_all_test2)
        return predict
        predict_proba = model.predict_proba(x_all_test2)
        y_true=[i[1] for i in y_all_test1]
        y_pred=[j[1] for j in predict_proba]
        auc=metrics.roc_auc_score(y_true=y_true,y_score=y_pred)

        return 1-auc
        #用作写报告哦############
        #以下是randomforest 的版本
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=20, max_depth=12, n_jobs=10)
        model.fit(x_all_train2, y_all_train1)
        predict_proba = model.predict_proba(x_all_test2)
        y_true=y_all_test1
        y_pred=[j[1] for j in predict_proba]
        auc=metrics.roc_auc_score(y_true=y_true,y_score=y_pred)
        del model

        return 1-auc

    args = (x_all_train1, x_all_test1)
    lb = [0 for i in range(len(x_all_test1[0]))]
    ub = [1 for i in range(len(x_all_test1[0]))]

    xopt, fopt = pso(weight, lb, ub, args=args, )
    numpy.set_printoptions(threshold=numpy.nan)
    numpy.savetxt('xopt_geneexpression.txt', xopt, fmt='%d') #最终只记录了1和0的消息，维度是17419的
    print(xopt,file=open("xopt_ge_print.txt","w"))

def pretreat_by_ga(data_all,target_all):
    import random
    import copy
    from deap import base
    from deap import creator
    from deap import tools
    import numpy as np
    def bool_selected_dims(container, n):
        assert type(n) == int
        twenty = int(n * 0.02)
        eighty = n - twenty
        one = [1 for i in range(twenty)]
        zero = [0 for j in range(eighty)]
        plus = one + zero
        random.shuffle(plus)
        copyplus = copy.deepcopy(plus)
        del plus
        return container(copyplus)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)  # 创建了两个变量，存储信息。

    toolbox = base.Toolbox()

    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)
    toolbox.register("attr_bool", random.randint, 0, 1)  # 包含了0,1的随机整数。

    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    # toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_bool, 46221)TODO：这是原本的语句，用在CNV上的
    toolbox.register("individual", bool_selected_dims, creator.Individual, 17419)

    # define the population to be a list of 'individual's
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized    注意！！！这里定义了我们的适应度fitness函数！！！
    # 只要返回一个值给我们的这个适应度函数啊！加入自己的机器学习方法；
    # 这里取名为evalOneMax是因为这里的适应度函数就是我们后面要用来评价的依据，evaluate。评估使用AUC。

    import numpy as np
    from yanru.pso_yanru import pso
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD
    from keras.optimizers import Adam
    import numpy
    from sklearn.externals import joblib
    from scipy import stats
    import keras.backend as K
    from keras.callbacks import ReduceLROnPlateau
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    import numpy as np
    from sklearn import metrics

    print('Now,loading ic50 data...')

    x_all_test1 = data_all[700:]
    x_all_train1 = data_all[:700]
    y_all_train1 = target_all[:700]
    y_all_test1 = target_all[700:]

    # positive_all = y_all_test1.count(1)
    # negative_all = y_all_test1.count(0)


    x_all_train1 = numpy.array(x_all_train1)
    y_all_train1 = numpy.array(y_all_train1)
    x_all_test1 = numpy.array(x_all_test1)

    # y_all_test1 = numpy.array(y_all_test1)

    # y_all_test1 = y_all_test1.tolist()

    # define the function: maybe the NN. and the x would be the weight of characters in all dims


    def evalOneMax(individual):
        x_all_train2 = x_all_train1 * individual
        x_all_test2 = x_all_test1 * individual
        x_all_train2 = x_all_train2.tolist()
        x_all_test2 = x_all_test2.tolist()
        indx = []

        for i, j in enumerate(individual):
            if j == 1:
                indx.append(i)
        print('index has been constructed')

        for i, j in enumerate(x_all_train2):
            # x_all_train2[i]=sorted(set(j),key=j.index)
            # print('done',i)
            x_all_train2[i] = [j[k] for k in indx]

        for i, j in enumerate(x_all_test1):
            # x_all_test2[i] = sorted(set(j), key=j.index)
            # print('done',i)
            x_all_test2[i] = [j[k] for k in indx]

        x_all_train2 = np.array(x_all_train2)
        x_all_test2 = np.array(x_all_test2)

        inputdim = len(x_all_train2[0])

        model = Sequential()
        model.add(Dense(inputdim // 2, input_dim=inputdim, init='uniform'))  # 输入层，28*28=784，最好附加初始化，用identity
        model.add(Activation('relu'))  # 激活函数是tanh(后面变成了relu因为对mnist的处理结果会好一些)
        # model.add(Dropout(0.2))  # 采用50%的dropout

        model.add(Dense(inputdim // 10))  # 隐层节点500个
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))

        model.add(Dense(inputdim // 100))  # 隐层节点500个
        model.add(Activation('relu'))

        model.add(Dense(inputdim // 1000))  # 隐层节点500个
        model.add(Activation('relu'))

        model.add(Dense(inputdim // 2000))  # 隐层节点500个
        model.add(Activation('relu'))

        model.add(Dense(2))  # 输出结果和药物对应的话只是一维的
        model.add(Activation('sigmoid'))  # 最后一层linear，因为是实际的结果

        # 设定学习率（lr）等参数
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, verbose=0, mode='min', epsilon=0.0001,
        #                               cooldown=0, min_lr=0)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
        # check_test = ModelCheckpoint('best_model_test0518.h5', monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
        model.fit(x_all_train2, y_all_train1, batch_size=15, epochs=50, shuffle=True, verbose=0)
        # model = load_model('best_model_test0518.h5')
        predict_proba = model.predict_proba(x_all_test2)
        predict_proba = predict_proba.reshape(-1)
        y_true = [i[1] for i in y_all_test1]
        y_pred = [j[1] for j in predict_proba]
        auc = metrics.roc_auc_score(y_true=y_true,y_score=y_pred)

        return auc,
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=20, max_depth=12, n_jobs=10)
        model.fit(x_all_train2, y_all_train1)
        predict_proba = model.predict_proba(x_all_test2)
        y_true = y_all_test1
        y_pred = [j[1] for j in predict_proba]
        auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        del model

        return auc,

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    # 这里的toolbox register语句的理解：注册了一个函数evaluae依据的是后面的evalOneMax 理通了!!!
    toolbox.register("evaluate", evalOneMax)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament,
                     tournsize=3)  # tournsize指的从individual中抽取tournsize个数出来，然后用于判断最大。没有输入的话默认是3，但是有输入的话就是输入的值。

    # ----------

    def main():
        random.seed(64)
        # hash(64)is used

        # random.seed方法的作用是给随机数对象一个种子值，用于产生随机序列。
        # 保证实验可重复

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=100)  # 定义了600个个体的种群！！！

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        #
        # NGEN  is the number of generations for which the
        #       evolution runs   进化运行的代数！果然，运行40代之后，就停止计算了
        CXPB, MUTPB, NGEN = 0.4, 0.4, 100

        print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))  # 这时候，pop的长度还是300呢

        # Begin the evolution
        for g in range(NGEN):
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))  # 实际上这里的select直接挑选了所有子代，找出最优的（这里是最大）
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            # sum2 = sum(x * x for x in fits)
            # std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        numpy.set_printoptions(threshold=numpy.nan)
        fw_ga = open('weight_for_ga.txt', 'w')
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values), file=fw_ga)
    main()


