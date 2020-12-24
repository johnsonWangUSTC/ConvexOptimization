'''
    Solving Problem 9 of Chapter 9 via EM Algorithm
'''

import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt

def main():

    F = np.array([162, 267, 271, 185, 111, 61, 27, 8, 3, 1])
    n = len(F)
    D = np.array(range(0, n))

    a = 0.3
    v1, v2 = 1, 2.5
    z = np.zeros(n)
    eps = 1e-15

    theta_dict = dict()
    theta_dict['a'] = []
    theta_dict['v1'] = []
    theta_dict['v2'] = []

    theta = np.array([a, v1, v2])
    theta_dict['a'].append(a)
    theta_dict['v1'].append(v1)
    theta_dict['v2'].append(v2)
    likelihood_list = []
    iteration = 0

    while iteration == 0 or np.linalg.norm(np.array([theta_dict['a'][iteration-1], theta_dict['v1'][iteration-1],
                                                     theta_dict['v2'][iteration-1]]) - theta) >= eps:

        for i in range(0, n):
            z[i] = a * np.exp(-v1) * pow(v1, i) / (a * np.exp(-v1) * pow(v1, i) + (1 - a) * np.exp(-v2) * pow(v2, i))
        a = np.sum(F * z) / np.sum(F)
        v1 = np.sum(F * D * z) / np.sum(F * z)
        v2 = np.sum(F * D * (np.ones(n)-z)) / np.sum(F * (np.ones(n)-z))
        theta = np.array([a, v1, v2])
        theta_dict['a'].append(a)
        theta_dict['v1'].append(v1)
        theta_dict['v2'].append(v2)

        iteration += 1

    print("Convergence in {} iterations with L2 update below {}".format(iteration, eps))
    print("a={}\nv1={}\nv2={}\n".format(a, v1, v2))

    for i in range(0, iteration + 1):
        likelihood = 1.0
        a, v1, v2 = theta_dict['a'][i], theta_dict['v1'][i], theta_dict['v2'][i]
        for j in range(0, n):
            likelihood *= pow((a * np.exp(-v1) * pow(v1, j) + (1 - a) * np.exp(-v2) * pow(v2, j)) / np.math.factorial(j), j)
        likelihood_list.append(likelihood)
    plt.plot(range(0, iteration+1), likelihood_list)
    print(likelihood_list[0], likelihood_list[1], likelihood_list[2])

    plt.show()



if __name__ == '__main__':
    main()
