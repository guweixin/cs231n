# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import time
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      ''' 使用L1距离'''
      ####distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      ''' 使用L2距离'''
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
   """ load all of cifar """
   xs = []
   ys = []
   for b in range(1,6):
     f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
     X, Y = load_CIFAR_batch(f)
     xs.append(X)#使变成行向量
     ys.append(Y)
   Xtr = np.concatenate(xs)
   Ytr = np.concatenate(ys)
   del X, Y
   Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
   return Xtr, Ytr, Xte, Yte



print("-----begin------"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

### 加载CIFAR-10图片，并分成4个数组：训练数据和标签，测试数据和标签
Xtr, Ytr, Xte, Yte = load_CIFAR10('./image_classesfication/data/cifar-10-batches-py/') # a magic function we provide

### 图片变成一维向量
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

### 最近邻分类器
nn = NearestNeighbor() # create a Nearest Neighbor classifier class

### 训练分类器
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
print("---train-done---"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ) 

### 预测输入的新数据的分类标签
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print("---predict-done--"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

### 准确率，它描述了我们预测正确的得分
print ('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))