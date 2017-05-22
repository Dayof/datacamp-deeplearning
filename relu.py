# rectified linear activation functions
# very high-perfomance networks
# without this function can be predicted a negative number
import numpy as np

def relu(input):
    return max(input, 0)

input_data = np.array([3, 5])
weights = { 'n0': np.array([2, 4]),
            'n1': np.array([4, -5]),
            'output': np.array([2, 7])}

n0_input = (input_data * weights['n0']).sum()
n0_output = relu(n0_input)

n1_input = (input_data * weights['n1']).sum()
n1_output = relu(n1_input)

hidden_layer_outputs = np.array([n0_output, n1_output])

model_output = (hidden_layer_outputs * weights['output']).sum()

print(model_output)
