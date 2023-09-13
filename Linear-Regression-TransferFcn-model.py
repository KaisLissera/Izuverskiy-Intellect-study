import numpy as np
import matplotlib.pyplot as plt

# global
ModelSize = 11
eta = 0.1
BatchSize = 10
noise = 0.4
trainSize = 100
testSize = 10
w = np.ones((1, ModelSize))

def Forward(inputs):
    #np.append(inputs, 1)
    output = np.sum(np.dot(w, inputs))
    return output

def Mse(outputs, target):
    mse = (1/ModelSize) * np.sum(np.square(target - outputs))
    return mse

def BackProp(trainInputs, trainTarget):
    for j in range(trainSize // BatchSize):
        batchLoss = 0
        grad = 0
        for i in range(BatchSize):
            input = trainInputs[:, j*BatchSize + i]
            target = trainTarget[j*BatchSize + i]

            output = Forward(input)
            error = output - target

            grad += 2 * input*error
            batchLoss += Mse(output, target)
        global w
        w -= eta * grad / BatchSize

def CreateDataset(signal):
    dataset = np.zeros((ModelSize, len(signal)))
    for i in range(len(signal)):
        dataset[ModelSize - 1, i] = 1
        for j in range(0, ModelSize - 1):
            if (i - j) < 0:
                dataset[j, i] = 0
            else:
                dataset[j, i] = signal[i - j]
    return dataset

#-----------------------------------------------------------------------
transferFcnParams = np.array((1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1))
transferFcnParams = transferFcnParams/5.5

def TransferFcn(signal):
    out = np.array(signal)
    for i in range(len(signal)):
        out[i] = 0
        for j in range(len(transferFcnParams)):
            if (i - j) < 0:
                break
            else:
                out[i] += transferFcnParams[j]*signal[i - j]
    return out

#-----------------------------------------------------------------------
time = np.linspace(0, 10, trainSize)
# Train step signal
signal = np.array(time) 
for i in range(len(time)):
    if(i > trainSize/2):
        signal[i] = 0
    else:
        signal[i] = 1

# Test step signal
signalTest = np.array(time)
for i in range(len(time)):
    if(i > trainSize/3):
        signalTest[i] = 0
    else:
        signalTest[i] = 1

signal = signal + np.random.uniform(-noise, noise, trainSize)
signalTest = signalTest + np.random.uniform(-0.1, 0.1, trainSize)

# Transfer Fcn
trainTarget = TransferFcn(signal)
testTarget = TransferFcn(signalTest)

# Creating training data
trainData = CreateDataset(signal)
# Creating test data
testData = CreateDataset(signalTest)

# Training
for i in range(20):
    BackProp(trainData, trainTarget)

# Testing
testOut = np.array(signalTest)
for i in range(len(signalTest)):
    testOut[i] = Forward(testData[:,i])

# Result
print(transferFcnParams)
print(w)

plt.plot(time, signal, 'b')
plt.plot(time, trainTarget, 'r')
plt.show()

plt.plot(time, signalTest, 'b')
plt.plot(time, testTarget, 'r')
plt.plot(time, testOut, 'g')
plt.show()