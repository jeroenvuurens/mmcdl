# author(s): jeroen
# based on an example from https://www.youtube.com/watch?v=S75EdAcXHKk
# sample noisy training points trX, trY around y = 2x, through (0,0)
# and train a gradient to fit a least squares line

import sys, numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# training points trX, trY
trX = np.linspace(-1,1,101);
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# input layer consisting of a training sample (X, Y)
X = T.scalar()
Y = T.scalar()

# gradient to be trained
w = theano.shared(np.asarray(0., dtype=theano.config.floatX))

# plot input and trained least squares line
plt.plot(trX, trY, 'ro')
fittedY = w.get_value() * trX
line, = plt.plot([], [])
line.set_xdata(trX)
line.set_ydata(fittedY)
plt.draw()
plt.show(block=False)

# predicted/output y based on input X and trained gradient w
y = X * w
# goal is to minimize least squares between predicted y and actual Y
# in this example, the mean square error is directly used as output layer
cost = T.mean(T.sqr(y - Y))
# gradient used to update the weight w
gradient = T.grad(cost, w)
# update step
updates = [[w, w - gradient * 0.01]]

# setup the training
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# draw the (optimized) function structure into an svg files
theano.printing.pydotprint(y, outfile="./pics/struct.svg", format='svg')

# train 20 epochs, using stochastic gradient descent over the training pairs trX, trY
for i in range(20):
    for x,y in zip(trX, trY):
        train(x,y)
    # redraw the fitted line
    line.set_ydata(w.get_value() * trX)
    plt.draw()

# the endvalue of w, which should be close to 2
print "final gradient", w.get_value()

# keeps the program alive to see the final solution
plt.show()
