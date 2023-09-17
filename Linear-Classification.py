import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# global
ModelSize = 2
eta = 0.1
lmbd = 0.05
BatchSize = 20
# noise = 0.9
trainSize = 400
# testSize = 10

# w = np.random.uniform(-1, 1, ModelSize)
# b = rnd.uniform(-1,1)
w = np.ones(ModelSize)
b = 1
print(w,b)

#-----------------------------------------------------------------------
def Forward(inputs):
    output = np.sum(np.dot(w, inputs)) + b
    return output

def Margin(output, target):
    if(target*output > 0):
        margin = target*output
    else:
        margin = -target*output
    return margin

def Activation(input):
    if -input > 0:
        return -input
    else:
        return 0

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
            error = Activation(output*target)

            dw += 2 * input*error 
            db += 2 * error
            batchLoss += Activation(output*target)
        global w, b
        b -= eta * db / BatchSize
        w -= (eta * dw / BatchSize + 2*eta*lmbd*np.sign(w)/BatchSize)
        loss[j] = batchLoss / BatchSize
    return loss

#-----------------------------------------------------------------------
x = np.zeros((2, trainSize//2))
x[0,:] = np.random.normal(-1, 0.5, trainSize//2)
x[1,:] = np.random.normal(1, 0.5, trainSize//2)

y = np.zeros((2, trainSize//2))
y[0,:] = np.random.normal(1, 0.5, trainSize//2)
y[1,:] = np.random.normal(-1, 0.5, trainSize//2)

# Dataset
Dataset = np.zeros((3, trainSize))

for i in range(trainSize//2):
    Dataset[0, i] = 1
    Dataset[1, i] = x[0, i]
    Dataset[2, i] = x[1, i]

for i in range(trainSize//2):
    Dataset[0, trainSize//2 + i] = -1
    Dataset[1, trainSize//2 + i] = y[0, i]
    Dataset[2, trainSize//2 + i] = y[1, i]

Dataset = Dataset.T
np.random.shuffle(Dataset)
Dataset = Dataset.T

# Training
loss = np.array(())
for i in range(200):
    loss_tmp = BackProp(Dataset[1:3, :], Dataset[0, :])
    loss = np.append(loss, loss_tmp)

print(w, b)
x_line = np.linspace(-2, 2, trainSize)
y_line = np.zeros(trainSize)
for i in range(trainSize):
    y_line[i] = -x_line[i]*w[0]/w[1] - b/w[1]

plt.plot(loss)
# plt.yscale('log')
plt.show()

plt.plot(x[0,:], x[1,:], 'bo')
plt.plot(y[0,:], y[1,:], 'ro')
plt.plot(x_line, y_line, 'g')
plt.show()