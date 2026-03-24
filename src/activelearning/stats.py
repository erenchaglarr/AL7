import numpy as np
import scipy.stats as stats

res_random = [0.788, 0.779, 0.801, 0.818, 0.817]
res_us = [0.842, 0.837, 0.837, 0.817, 0.819]
res_three = [0.819, 0.829, 0.826, 0.829, 0.811]
res_five = [0.837, 0.829, 0.831, 0.833, 0.816]
res_ten = [0.810, 0.839, 0.813, 0.824, 0.830]
res_twenty = [0.808, 0.830, 0.837, 0.841, 0.828]


print(f"Stats for Random Sampling: Mean {np.mean(res_random):.3f}, Std: + {np.std(res_random):.3f}")
print(f"Stats for Uncertainty Sampling: Mean {np.mean(res_us):.3f}, Std: + {np.std(res_us):.3f}")
print(f"Stats for three query members: Mean {np.mean(res_three):.3f}, Std: + {np.std(res_three):.3f}")
print(f"Stats for five query members: Mean {np.mean(res_five):.3f}, Std: + {np.std(res_five):.3f}")
print(f"Stats for ten query members: Mean {np.mean(res_ten):.3f}, Std: + {np.std(res_ten):.3f}")
print(f"Stats for twenty query members: Mean {np.mean(res_twenty):.3f}, Std: + {np.std(res_twenty):.3f}")

