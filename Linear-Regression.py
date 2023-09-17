import numpy as np
import matplotlib.pyplot as plt

# global
ModelSize = 1
eta = 0.5
BatchSize = 10
noise = 0.2
trainSize = 300
testSize = 10
w = np.zeros(ModelSize)
b = 0

#-----------------------------------------------------------------------
def Forward(inputs):
    output = np.sum(np.dot(w, inputs)) + b
    return output

def Mse(outputs, target):
    mse = (1/ModelSize) * np.sum(np.square(target - outputs))
    return mse

def BackProp(trainInputs, trainTarget):
    for j in range(trainSize // BatchSize):
        # batchLoss = 0
        dw = 0
        db = 0
        for i in range(BatchSize):
            input = trainInputs[:, j*BatchSize + i]
            target = trainTarget[j*BatchSize + i]

            output = Forward(input)
            error = output - target

            dw += 2 * input*error
            db += 2 * error
            # batchLoss += Mse(output, target)
        global w, b
        b -= eta * db / BatchSize
        w -= eta * dw / BatchSize

#-----------------------------------------------------------------------
def CreateRegressionDataset(signal):
    dataset = np.zeros((ModelSize, len(signal)))
    for i in range(len(signal)):
        for j in range(0, ModelSize):
            if (i - j) < 0:
                dataset[j, i] = 0
            else:
                dataset[j, i] = signal[i - j]
    return dataset

def LagPlot(data):
    lagSize = 1
    x_lag = np.zeros(len(data) - lagSize)
    y_lag = np.zeros(len(data) - lagSize)
    for i in range(len(data) - lagSize):
        x_lag[i] = data[i]
        y_lag[i] = data[i + lagSize]
    plt.plot(x_lag, y_lag, 'ro')
    plt.show()

#-----------------------------------------------------------------------
x = np.linspace(0, 1, trainSize)
y =  -0 + x**1 + np.random.uniform(-noise, noise, trainSize)
# LagPlot(y)

for i in range(10):
    BackProp(x.reshape(1, trainSize), y)

print("b = ", b," w = ", w)

y_test = np.array(x)
for i in range(len(y_test)):
    y_test[i] = Forward(x[i])

plt.plot(x, y, 'b')
plt.plot(x, y_test, 'r')
plt.show()