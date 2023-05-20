import matplotlib.pyplot as plt
from timeit import default_timer as timer
import random
import numpy as np

def main():
    #test_transpose()
    test_mult()

def test_mult():
    n_limit = 100
    test_size = 100
    times = []
    sizes = []
    theoretical_times = []
    n_cubed = []
    for i in range(1, n_limit):
        elapsed_time = 0
        for j in range(1, test_size):
            a = np.random.randint(1, 10, size=(i, i))
            b = np.random.randint(1, 10, size=(i, i))
            c = np.random.randint(1, 10, size=(i, i))
            start = timer()
            Mult(a, b, c, i)
            end = timer()
            elapsed_time += end - start
        times.append(elapsed_time / test_size)
        sizes.append(i)
        theoretical_times.append(2 * (i ** 3) + 3 * (i**2) + 2*i + 2)
        n_cubed.append(i**3)

    c = times[-1] / theoretical_times[-1]
    theoretical_times = [i * c for i in theoretical_times]
    c = times[-1] / n_cubed[-1]
    n_cubed = [i * c for i in n_cubed]

    plt.title("Matrix Multiplication Run Time vs Input Size")
    plt.xlabel("Matrix Size [n]")
    plt.ylabel("Time [s]")
    plt.plot(sizes,times,label="Measured Times")
    plt.plot(sizes,theoretical_times,label="Frequency Count")
    plt.plot(sizes,n_cubed,label="Big Theta")
    plt.legend()
    plt.show()

def test_transpose():
    n_limit = 100
    test_size = 100
    times = []
    sizes = []
    theoretical_times = []
    n_squared = []
    for i in range(1,n_limit):
        elapsed_time = 0
        for j in range(1,test_size):
            a = np.random.randint(1, 10, size=(i, i))
            start = timer()
            Transpose(a, i)
            end = timer()
            elapsed_time += end - start
        times.append(elapsed_time/test_size)
        sizes.append(i)
        theoretical_times.append(2*(i**2) + 4*i + 2)
        n_squared.append(i**2)

    c = times[-1]/theoretical_times[-1]
    theoretical_times = [i*c for i in theoretical_times]
    c = times[-1] / n_squared[-1]
    n_squared = [i * c for i in n_squared]

    plt.title("Transpose Run Time vs Input Size")
    plt.xlabel("Matrix Size [n]")
    plt.ylabel("Time [s]")
    plt.plot(sizes,times,label="Measured Times")
    plt.plot(sizes,theoretical_times,label="Frequency Count")
    plt.plot(sizes,n_squared,label="Big Theta")
    plt.legend()
    plt.show()

def Transpose(a, n):
    for i in range(0,n):
        for j in range(i,n):
            t = a[i][j]
            a[i][j] = a[j][i]
            a[j][i] = t
    return a

def Mult(a, b, c, n):
    for i in range(0,n):
        for j in range(0,n):
            c[i][j] = 0
            for k in range(0,n):
                c[i][j] += a[i][k]*b[k][j]
    return c

main()