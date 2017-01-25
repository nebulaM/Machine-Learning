import numpy as np

def foo(N):
    for i in range(N):
        print("Hello!")

def bar(N):
    x = np.zeros(N)
    return x

def bar(N):
    x = np.zeros(N)
    x += 1000
    return x

def bat(N):
    x = np.zeros(1000)
    x = x * N
    return x
