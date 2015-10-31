#!/usr/bin/env python
#coding=gbk
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

I modified the code to binary logistic regression using auc as evaluation method

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy
import numpy as np 

import theano
theano.config.openmp = True
import theano.tensor as T

from sklearn.metrics import roc_auc_score
from theano import gof, config

from gflags import *

DEFINE_float('learning_rate', 0.13, '')
DEFINE_integer('batch_size', 600, '')
DEFINE_integer('n_epochs', 1000, '')
DEFINE_float('min_improvement', 0.0001, '')
DEFINE_integer('iter', 10000, '')
    
class RocAucScoreOp(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.
    Parameters
    ----------
    name : str, optional (default 'roc_auc')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='roc_auc', use_c_code=theano.config.cxx):
        super(RocAucScoreOp, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate ROC AUC score.
        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted class labels or probabilities for positive class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.scalar(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate ROC AUC score.
        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        if roc_auc_score is None:
            raise RuntimeError("Could not import from sklearn.")
        y_true, y_score = inputs
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = np.nan
        output_storage[0][0] = theano._asarray(roc_auc, dtype=config.floatX)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
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
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
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
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
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
        #[]is for correctly index to the column of label index
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

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
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def auc(self, y):
        return RocAucScoreOp('roc_auc')(y, self.p_y_given_x[:,1])
        #return RocAucScoreOp('roc_auc')(y, self.y_pred)


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset 
    '''
    cache_file = ''
    if dataset.endswith('.txt'):
        cache_file = dataset.replace('.txt', '.pkl')
    else:
        cache_file = dataset + '.pkl'

    print '... loading data:',dataset
    dataset_x = []
    dataset_y = []

    print cache_file, os.path.isfile(cache_file)
    if os.path.isfile(cache_file):
        dataset_x, dataset_y= cPickle.load(open(cache_file, 'rb'))
    else:
        isfirst = True
        nrows = 0
        for line in open(dataset):
            if isfirst:
                isfirst = False
                continue
            nrows += 1
            if nrows % 10000 == 0:
                print nrows
            l = line.strip().split()
            label_idx = 0
            if l[0].startswith('_'):
                label_idx = 1
            dataset_y.append(float(l[label_idx]))
            dataset_x.append([float(x) for x in l[label_idx + 1:]])

        cPickle.dump((dataset_x, dataset_y), open(cache_file, 'wb'))

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y) 

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
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
        return shared_x, T.cast(shared_y, 'int32')

    data_set_x, data_set_y = shared_dataset(dataset_x, dataset_y)

    return data_set_x, data_set_y


def sgd_optimization(train_set, valid_set, test_set,
    learning_rate=0.13, n_epochs=1000, batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model
    """
    print 'loading ', train_set, ' for train'
    train_set_x, train_set_y = load_data(train_set)
    print 'loading ', valid_set, ' for valid'
    valid_set_x, valid_set_y = load_data(valid_set)
    print 'loading ', test_set, ' for test'
    if test_set != valid_set:
        test_set_x, test_set_y = load_data(test_set)
    else:
        test_set_x, test_set_y = valid_set_x, valid_set_y

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "train_set_x size:",train_set_x.get_value(borrow=True).shape[0]
    print "batch_size:",batch_size
    print "n_train_batches:",n_train_batches
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    total_dim = train_set_x.get_value(borrow=True).shape[1]
    #classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    classifier = LogisticRegression(input=x, n_in=total_dim, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

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

    validate_auc = theano.function(
        inputs=[],
        outputs=classifier.auc(y),
        givens={
            x: valid_set_x,
            y: valid_set_y
        }
    )

    test_auc = theano.function(
        inputs=[],
        outputs=classifier.auc(y),
        givens={
            x: test_set_x,
            y: test_set_y
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
    print '... training the model'
    # early-stopping parameters
    patience = FLAGS.iter  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    print "n_train_batches:",n_train_batches
    print "validation_frequency:",validation_frequency

    best_validation_loss = numpy.inf
    best_auc = 0
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
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
                auc_values = [validate_auc()]
                auc = numpy.mean(auc_values)
                print "current valid auc: ",auc, " best auc: ", best_auc, " imporve: ", auc - best_auc, " significant?: ", auc - best_auc > FLAGS.min_improvement
                #print validate_auc(0)

                if auc > best_auc:
                    if auc - best_auc > FLAGS.min_improvement:
                        print 'before patience:',patience,' iter:',iter
                        patience = max(patience, iter * patience_increase)
                        print 'after patience:',patience
                    best_auc = auc
                    auc_values = [test_auc()]
                    testauc = numpy.mean(auc_values)
                    print "test auc: ",testauc 
                    cPickle.dump(classifier, open('best_model.pkl', 'wb'))

            if patience <= iter:
                done_looping = True
                print "patience:",patience,"iter:",iter,"done_looping:",done_looping
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print 'best valid auc is ',best_auc
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


#def predict(dataset_ = 'mnist.pkl.gz'):
#    """
#    An example of how to load a trained model and use it
#    to predict labels.
#    """
#
#    # load the saved model
#    classifier = cPickle.load(open('best_model.pkl'))
#
#    # compile a predictor function
#    predict_model = theano.function(
#        inputs=[classifier.input],
#        outputs=classifier.y_pred)
#
#    # We can test it on some examples from test test
#    dataset=dataset_
#    datasets = load_data(dataset)
#    test_set_x, test_set_y = datasets[2]
#    test_set_x = test_set_x.get_value()
#
#    predicted_values = predict_model(test_set_x[:10])
#    print ("Predicted values for the first 10 examples in test set:")
#    print predicted_values
#    print "true value of first 10:"
#    #test_set_y = test_set_.get_value()
#    #print test_set_y.eval()[:10]


if __name__ == '__main__':
    argv = sys.argv
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError, e:
        print '%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)
    trainset_ = argv[1]
    validset_ = argv[2]
    testset_ = argv[3]

    sgd_optimization(train_set = trainset_, valid_set = validset_, test_set = testset_, 
        learning_rate = FLAGS.learning_rate, n_epochs = FLAGS.n_epochs, batch_size = FLAGS.batch_size)
    #predict(dataset_)
