"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import math
from sklearn.externals import joblib


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,activation):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)#TODO:i have changed all W and b into ones form.used to be zeros
        self.W = theano.shared(
            value=numpy.ones(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        #self.p_y_given_x = activation(T.dot(input, self.W) + self.b)#this algebra should be changed into others,eg relu.hasn't changed.relu used to be 'softmax'
        p_y_given_x=T.dot(input, self.W) + self.b
        self.p_y_given_x=(p_y_given_x if activation is None
                          else activation(p_y_given_x))#this is my addition
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)  #if in my model, i need to change it into linear
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #todo: adding some new elements!!!!!!!!!!!
        '''nrow=len(self.p_y_given_x.eval())
        ncol=len(self.p_y_given_x[0].eval())
        p_y_given_x_mean=[]
        for i in self.p_y_given_x:
            p_y_given_x_mean.append(sum(i)/ncol)
        p_y_given_x_mean=numpy.array(p_y_given_x_mean)
        return -T.sqrt(T.mean(((y - p_y_given_x_mean) ** 2)))'''
        return T.sqrt(T.mean(((y - self.y_pred) ** 2)))
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2
    '''def mse(self,y):
        return -T.sqrt(T.mean(((y-self.p_y_given_x)**2)[T.arange(y.shape[0]),y]))#this is the place i need to change'''

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        '''if y.dtype.startswith('int'):#todo here we should enlarege the type.for example, float
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))#todo we may need to hide this return. for example, here replaced with pass
        else:
            raise NotImplementedError()'''


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    '''actually jsdhflasdjklfsdkljflksjadlkfjsdlkjfkdslaksldhjsghak
    i don't need the following lines(some),i can contruct my data. and save the shared_data part
    but still need to store my data in a dataset name datasets.it means that i need to creat a dataset then give value to datasets'''
    # Download the MNIST dataset if it is not present
    '''data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)'''

    '''print('Now,loading ic50 data...')#this is for loading data in slow way in pc in order not raise memory error
    f_notuseful = open(dataset, 'r')
    f_notuseful.close()
    f_ic50 = open('findout_ic50alldict.txt', 'r')
    fread = f_ic50.read()
    ic50_dict = eval(fread)
    f_ic50.close()

    f_geneExp = open('geneExp_dict.txt', 'r')
    # fw=open('trybb.txt','a')
    line = f_geneExp.readline()
    data = []
    target = []
    data_dict = {}
    print('Organizing data_dict...')
    while line:
        linetry = line.split('  	 ')
        data_dict[linetry[0]] = eval(linetry[1])
        line = f_geneExp.readline()
    f_geneExp.close()

    print('data_dict finished.')
    print('Now, loading data and target...')
    for keys in ic50_dict['1']:
        if keys in data_dict:
            target.append(ic50_dict['1'][keys])
            data.append(data_dict[keys])
        else:
            pass
    targetraw=[]
    for i in target:
        j=round(math.exp(i)*100)
        targetraw.append(j)
    target=[]
    test_set = (data[300:], targetraw[300:])
    valid_set = (data[200:300], targetraw[200:300])
    train_set = (data[:200], targetraw[:200])
    data=[]
    targetraw=[]

    #print('Origin data and target complete. Now begin to calculate.')'''


    '''print('... loading data')#this is the version of loading test data on PC
    f=open(dataset,'r')
    fread=f.readline()
    data=eval(fread)
    fread=f.readline()
    target=eval(fread)
    test_set=(data[25:],target[25:])
    valid_set=(data[20:25],target[20:25])
    train_set=(data[:20],target[:20])'''

    '''print('Now,loading ic50 data...')#this is the version of loading data on public computer
    f_notuseful=open(dataset,'r')
    f_notuseful.close()
    f_ic50 = open('targetfortest.txt', 'r')
    fread = f_ic50.read()
    ic50_dict = eval(fread)
    target = ic50_dict
    f_ic50.close()

    f_geneExp = open('datafortest.txt', 'r')
    # fw=open('trybb.txt','a')
    datafortest = f_geneExp.read()
    data = eval(datafortest)
    f_geneExp.close()
    targetraw = []
    for i in target:
        j = round(math.exp(i) * 100)
        targetraw.append(j)
    target = []
    test_set = (data[300:], targetraw[300:])
    valid_set = (data[200:300], targetraw[200:300])
    train_set = (data[:200], targetraw[:200])
    data = []
    targetraw = []
    print('data_dict finished.')
    print('Now, loading data and target...')
    print('Origin data and target complete. Now begin to calculate.')'''
    print('Now,loading ic50 data...')  # this is the version of loading data on public computer
    f_notuseful = open(dataset, 'r')
    f_notuseful.close()
    fm = open('abalone_m.txt', 'r')
    fread = fm.readline()
    data = eval(fread)
    fread = fm.readline()
    target = eval(fread)
    fm.close()

    test_set = (data[1300:], target[1300:])
    valid_set = (data[1000:1300], target[1000:1300])
    train_set = (data[:1000], target[:1000])
    '''f_notuseful=open(dataset, 'r')
    f_notuseful.close()
    f_data = open('data_drug203.txt', 'r')
    fread = f_data.read()
    data_origin = eval(fread)
    #data_origin = data_origin[850:]
    f_target = open('target_drug203.txt', 'r')
    fread1 = f_target.read()
    target = eval(fread1)
    #target = target[850:]
    V = numpy.array(data_origin)

    pca = joblib.load('model_for_pca.m')
    data = pca.transform(V)
    numpy.set_printoptions(threshold=numpy.nan)
    data = numpy.array(data)
    target = numpy.array(target)
    test_set = (data[850:], target[850:])
    valid_set = (data[800:850], target[800:850])
    train_set = (data[:800], target[:800])'''

    # Load the dataset
    '''with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)'''
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):#it means that i need to create dataset=[(train_set),(),()]
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x,shared_y #TODO: T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)#first shared second put my data in one dataset

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=10,
                           dataset='/home/cuckoo/data/zengyanru/datafortest.txt',#used to be mnist.pkl.gz
                           batch_size=30):#change 'mnist.pkl.gz' into 'dataset',remember to change it ok? for my example.
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=7, n_out=300)#in my own database,i need to change input dims and output dims.used to be  n_in=28 * 28, n_out=10

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)#used to be 'negative_log_likelihood'

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model_forlsgd.pkl', 'wb') as f:
                        pickle.dump(classifier, f)#need to add this in dbn

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    '''print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)'''


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    f=open('best_model_forlsgd.pkl','rb')
    lsgd =pickle.load(f)

    # compile a predictor function
    predict_model = theano.function(
        inputs=[lsgd.input],
        outputs=lsgd.y_pred)

    # We can test it on some examples from test test
    dataset='abalone_m.txt'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    #sgd_optimization_mnist()
    predict()