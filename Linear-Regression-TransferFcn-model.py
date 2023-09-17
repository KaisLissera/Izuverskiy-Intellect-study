import numpy as np
import matplotlib.pyplot as plt

# global
ModelSize = 10
eta = 0.05
BatchSize = 5
noise = 0.1
trainSize = 200
testSize = 10
w = np.zeros(ModelSize)
b = 0

def Forward(inputs):
    output = np.sum(np.dot(w, inputs)) + b
    return output

def Mse(outputs, target):
    mse = np.sum(np.square(target - outputs))
    return mse

def BackProp(trainInputs, trainTarget):
    loss = np.zeros(trainSize // BatchSize)
    for j in range(trainSize // BatchSize):
        batchLoss = 0
        dw = 0
        db = 0
        for i in range(BatchSize):
            input = trainInputs[:, j*BatchSize + i]
            target = trainTarget[j*BatchSize + i]

            output = Forward(input)
            error = output - target

            dw += 2 * input*error
            db += 2 * error
            batchLoss += Mse(output, target)
        global w, b
        b -= eta * db / BatchSize
        w -= eta * dw / BatchSize
        loss[j] = batchLoss / BatchSize
    return loss

def CreateDataset(signal):
    dataset = np.zeros((ModelSize, len(signal)))
    for i in range(len(signal)):
        for j in range(0, ModelSize):
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

def Step(time, position = 0.5):
    signal = np.array(time) 
    for i in range(len(time)):
        if(i > len(time)*position):
            signal[i] = 0
        else:
            signal[i] = 1
    return signal

#-----------------------------------------------------------------------
time = np.linspace(0, 10, trainSize)
# Train step signal
signal = Step(time)
# Test step signal
signalTest = Step(time, 0.3)

signal = signal + np.random.uniform(-noise, noise, trainSize)
signalTest = signalTest + np.random.uniform(-0.1, 0.1, trainSize)

# Transfer Fcn
trainTarget = TransferFcn(signal)
# trainTarget = trainTarget + np.random.uniform(-0.2, 0.2, trainTarget.shape)
testTarget = TransferFcn(signalTest)

# Creating training data
trainData = CreateDataset(signal)
# Creating test data
testData = CreateDataset(signalTest)

# Training
loss = np.array(())
for i in range(10):
    loss_tmp = BackProp(trainData, trainTarget)
    loss = np.append(loss, loss_tmp)
plt.plot(loss)
plt.yscale('log')
plt.show()

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