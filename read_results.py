import pickle

filename = f'benchmark_CIFAR10/results_pkl/resnet8final_res_80_70_Pc8_Pm8_seed1.pkl'
filename2 = f'benchmark_CIFAR10/results_pkl/resnet8pareto_conf_80_70_Pc8_Pm8_seed1.pkl' 
filename3 = f'benchmark_CIFAR10/results_pkl/resnet8pareto_res_80_70_Pc8_Pm8_seed1.pkl' 

# Open the pickle file in read-binary mode

with open(filename, 'rb') as file:
    data = pickle.load(file)
print(f"final res: {data}")

with open(filename2, 'rb') as file:
    data2 = pickle.load(file)
print(f"conf : {data2}")

with open(filename3, 'rb') as file:
    data3 = pickle.load(file)
print(f"pareto res: {data3}")
