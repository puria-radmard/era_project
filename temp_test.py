import numpy as np
from matplotlib import pyplot as plt

p_truth = np.linspace(0, 1.0, 100)

for temp in np.arange(10):
    p_truth_temp = np.power(p_truth, 1.0 / temp)
    p_lie_temp = np.power(1 - p_truth, 1.0 / temp)
    p_truth_tempered = p_truth_temp / (p_truth_temp + p_lie_temp)
    plt.plot(p_truth, p_truth_tempered, label = temp)

plt.legend()

plt.savefig('temp_test.png')
