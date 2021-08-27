# 1st time run
# 1. Need to copy all c headers from ".\venv\Lib\site-packages\numpy\core\include\numpy" to ".\venv\include"
# 2. python setup.py install
# 3. python test.py

import numpy as np
import myModule

num_row = 2
num_col = 3

input_1 = np.round(np.random.rand(num_row,num_col)*5,2).astype("double")
print("1st array: ")
print(input_1)

input_2 = np.round(np.random.rand(num_row,num_col)*5,2).astype("double")
print("2nd array: ")
print(input_2)

output = np.zeros((num_row,num_col), dtype="double")
myModule.ext_matadd(num_row,num_col,input_1, input_2, output)
print("Output array: ")
print(output)