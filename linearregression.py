import sys, numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

trX = np.linspace(-1,1,101);
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

plt.plot(trX, trY, 'ro')

X = T.scalar()
Y = T.scalar()

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = X * w
fittedY = w.get_value() * trX

line, = plt.plot([], [])
line.set_xdata(trX)
line.set_ydata(fittedY)
plt.draw()
plt.show(block=False)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost, w)
updates = [[w, w - gradient * 0.01]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

v = theano.tensor.vector()

theano.printing.pydotprint(y, outfile="./pics/struct.svg", format='svg')

for i in range(20):
    for x,y in zip(trX, trY):
        train(x,y)
    line.set_ydata(w.get_value() * trX)
    plt.draw()

# the endvalue of w, which should be close to 2
print "final gradient", w.get_value()

# keeps the program alive to see the final solution
plt.show()


