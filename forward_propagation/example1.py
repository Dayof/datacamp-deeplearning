import numpy as np

input_data = np.array([2, 3])
weights = { 'n0': np.array([1, 1]),
            'n1': np.array([-1, 1]),
            'output': np.array([2, -1])}
n0_value = (input_data * weights['n0']).sum()
n1_value = (input_data * weights['n1']).sum()

hlv = np.array([n0_value, n1_value])
output = (hlv * weights['output']).sum()

print(hlv, output)
