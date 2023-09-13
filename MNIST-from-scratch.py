import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt

def LoadDataCsv(fileName, printHead = False):
    data = pd.read_csv(fileName)
    if(printHead):
        print(data.head())
    #
    data = np.array(data)
    np.random.shuffle(data)
    return data

def GetDataLable(data, line):
    return data[line][0]

def GetDataInput(data, line):
    return np.array(data[line][1:785]).reshape(784,1)

def PlotData(data, digitsInRow):
    fig, ax = plt.subplots(digitsInRow,digitsInRow)
    for i in range(0, digitsInRow**2):
        example = GetDataInput(data, i).reshape(28,28)
        ax[i // digitsInRow, i % digitsInRow].imshow(example, cmap='gray')
        ax[i // digitsInRow, i % digitsInRow].axis('off')
        ax[i // digitsInRow, i % digitsInRow].set_title('Lable: ' + str(GetDataLable(data, i)))
    plt.show()

#########################################################################
class QuadraticCost:
    @staticmethod
    def Cost(output, lable):
        y = np.zeros((10,1))
        y[lable][0] = 1
        return np.sum((output - y)**2)
    
    @staticmethod
    def Delta(output, z, lable):
        y = np.zeros((10,1))
        y[lable][0] = 1
        return np.array((output - y) * SigmoidDerivative(z))
    
class CrossEntropyCost:
    @staticmethod
    def Cost(output, lable):
        y = np.zeros((10,1))
        y[lable][0] = 1
        return np.sum(-y*np.log(output) - (1 - y)*np.log(1 - output))
    
    @staticmethod
    def Delta(a, z, lable):
        y = np.zeros((10,1))
        y[lable][0] = 1
        return a - y
    
#########################################################################
class Network:
    def __init__(self, learningRate, batchSize, lmbd = 0, reg =''):
        layer2 = 200
        self.w1 = np.random.randn(layer2,784) / m.sqrt(784)
        self.b1 = np.random.randn(layer2,1)
        self.w2 = np.random.randn(10,layer2) / m.sqrt(layer2)
        self.b2 = np.random.randn(10,1)

        self.eta = learningRate
        self.batchSize = batchSize
        self.lmbd = lmbd
        self.reg = reg

    def Forward(self, inputs):
        a1 = Sigmoid(np.dot(self.w1, inputs) + self.b1)
        a2 = Sigmoid(np.dot(self.w2, a1) + self.b2)
        return a2
    
    def BackPropogation(self, inputs, lable):
        # Forward
        z1 = np.dot(self.w1, inputs) + self.b1
        a1 = Sigmoid(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = Sigmoid(z2)
        # Back
        #dlt2 = QuadraticCost.Delta(a2, z2, lable)
        dlt2 = CrossEntropyCost.Delta(a2, z2, lable)
        db2 = dlt2
        dw2 = np.dot(dlt2, a1.T)

        dlt1 = np.dot(self.w2.T, dlt2) * SigmoidDerivative(z1)
        db1 = dlt1
        dw1 = np.dot(dlt1, inputs.T)
        
        return db1, db2, dw1, dw2
    
    def Train(self, trainData):
        np.random.shuffle(trainData)
        inputs = np.array(trainData[:,1:])
        lables = np.array(trainData[:,0])
        #
        trainDataLen = 60000
        trainCost = []
        for i in range(0, trainDataLen // self.batchSize):
            db1 = 0
            db2 = 0
            dw1 = 0
            dw2 = 0
            cost = 0
            for j in range(0, self.batchSize):
                pos = i*self.batchSize + j
                input = inputs[pos,:].reshape(784,1)
                lable = lables[pos]
                input = np.divide(input, 255)
                _db1, _db2, _dw1, _dw2 = Net.BackPropogation(input, lable)
                db1 = db1 + _db1
                db2 = db2 + _db2
                dw1 = dw1 + _dw1
                dw2 = dw2 + _dw2
                #cost = cost + QuadraticCost.Cost(Net.Forward(input), lable)
                cost = cost + CrossEntropyCost.Cost(Net.Forward(input), lable)
            trainCost.append(cost / self.batchSize)
            self.b1 = self.b1 - (self.eta * db1 / self.batchSize)
            self.b2 = self.b2 - (self.eta * db2 / self.batchSize)
            if(self.reg == 'L1'):
                self.w1 = self.w1 - self.eta*self.lmbd*np.sign(self.w1)/trainDataLen - (self.eta * dw1 / self.batchSize)
                self.w2 = self.w2 - self.eta*self.lmbd*np.sign(self.w2)/trainDataLen - (self.eta * dw2 / self.batchSize)
            elif(self.reg == 'L2'):
                self.w1 = self.w1*(1 - self.eta*self.lmbd/trainDataLen) - (self.eta * dw1 / self.batchSize)
                self.w2 = self.w2*(1 - self.eta*self.lmbd/trainDataLen) - (self.eta * dw2 / self.batchSize)
            else:
                self.w1 = self.w1 - (self.eta * dw1 / self.batchSize)
                self.w2 = self.w2 - (self.eta * dw2 / self.batchSize)
                
        return trainCost

    def LableToArray(self, lable):
        array = np.zeros((10,1))
        array[lable][0] = 1
        return array

    def Test(self, testData):
        testDataLen = 10000
        errors = 0
        for i in range(0,testDataLen):
            input = GetDataInput(testData, i)
            input = np.divide(input, 255)
            lable = GetDataLable(testData, i)
            if(np.argmax(Net.Forward(input)) != lable):
                errors = errors + 1
        return ((testDataLen - errors)/testDataLen)*100
    
######################################################################### 
  
def Sigmoid(z):
    return np.array(1 / (1 + np.exp(-z)))

def SigmoidDerivative(z):
    return np.array(Sigmoid(z)*(1 - Sigmoid(z)))

def Softmax(z):
    data = z
    for i in range(10):
        data[i] = np.exp(z[i]) / np.sum(np.exp(z))
    return np.array(data)
#########################################################################

trainData = LoadDataCsv('mnist_train.csv')
testData = LoadDataCsv('mnist_test.csv')
#PlotData(trainData,3)

Net = Network(0.1, 10, 10, reg = 'L2')

trainCost = []
test = []
#trainCost = Net.Train(trainData)
for i in range(0,20):
    trainCost = np.concatenate((trainCost, Net.Train(trainData)))
    print('Epoch: ' + str(i + 1) + ' Test error: ' + str(Net.Test(testData)))
    print('Epoch: ' + str(i + 1) + ' Train error: ' + str(Net.Test(trainData)))
    print('')

plt.plot(np.log(trainCost))
#plt.plot(test)
plt.show()