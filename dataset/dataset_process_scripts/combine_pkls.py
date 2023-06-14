# combine multiple pickle files into a single one

import pickle
import sys


output_name = sys.argv[1]
input_names = sys.argv[2:]

if output_name in ["-h", "--help"]:
    print("Usage: combine_pkls.py [output_name] [input_file1] [input_file2] ...")
    exit()

combined_result = None
for idx, input_file in enumerate(input_names):
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    if idx==0:
        combined_result = data
        if type(data) is list:
            data_type = "list"
        elif type(data) is dict:
            data_type = "dict"
    else:
        if data_type == "list":
            combined_result.extend(data)
        elif data_type == "dict":
            combined_result.update(data)
            
with open(output_name, "wb") as f:
    pickle.dump(combined_result, f)
        