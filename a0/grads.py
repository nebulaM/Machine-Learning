import numpy as np
from scipy.optimize import approx_fprime


def example(x):
    return np.sum(x**2)


def example_grad(x):
    return 2.0*x


def foo(x):
    return np.prod(x)


def foo_grad(x):
    #pass  # "pass" just means "do nothing"; it's here to avoid a "compile error"
    # TODO: FILL THIS IN
	re=[]
	prod=np.prod(x)
	for i in range(0,len(x)):
		re.append(prod/x[i])
	return re
		

def bar(x):
    result = 0
    for x_i in x:  # iterate through the elements of x
        result = result * x_i
    return result


def bar_grad(x):
    #pass
    # TODO: FILL THIS IN
	re=[]
	for i in range(0,len(x)):
		re.append(0)
	return re

# here is some code to test your answers
# below we test out example_grad using scipy.optimize.approx_fprime,
# which approximates gradients.
# if you want, you can use this to test out your foo_grad and bar_grad
x0 = np.random.rand(5) # take a random x-vector just for testing
diff = approx_fprime(x0, example, 1e-4)  # don't worry about the 1e-4 for now
print("x0 is %s"%x0)

print("My gradient     : %s" % example_grad(x0))
print("Scipy's gradient: %s" % diff)
# if you run this file with "python grads.py" the code above will run.
print("My product is: %s" % foo_grad(x0))
print("Scipy's product is : %s" % approx_fprime(x0,foo,1e-4))
print("My 0 is : %s" %bar_grad(x0))
print("Scipy's 0 is: %s" %approx_fprime(x0,bar,1e-4))
