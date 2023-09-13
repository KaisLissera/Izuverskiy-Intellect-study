import numpy as np
import matplotlib.pyplot as plt

# global
ModelSize = 2
eta = 0.1
BatchSize = 10
noise = 0.1
trainSize = 100
testSize = 10
w = np.ones((1,ModelSize))

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

#-----------------------------------------------------------------------

x0 = np.ones((1, trainSize))
x1 = np.linspace(0, 1, trainSize)
x = np.append(x0, x1).reshape(ModelSize, trainSize)

y =  1 + x1**2 + np.random.uniform(-noise, noise, trainSize)

for i in range(1):
    BackProp(x, y)

x0_t = np.ones((1, testSize))
x1_t = np.linspace(0, 1, testSize)
x_t = np.append(x0_t, x1_t).reshape(ModelSize, testSize)

y_test = x1_t
for i in range(testSize):
    y_test[i] = Forward(x_t[:,i])

plt.plot(x[1,:], y, 'bo')
plt.plot(x_t[1,:], y_test, 'ro')
plt.show()