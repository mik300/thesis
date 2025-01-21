import pickle
import torch

# filename = f'benchmark_CIFAR10/results_pkl/resnet8_backup_1_70_Pc8_Pm8_seed1.pkl' 
# filename = f'benchmark_CIFAR10/results_pkl/resnet8final_res_80_70_Pc8_Pm8_seed1.pkl'
# filename = f'benchmark_CIFAR10/results_pkl/resnet8pareto_conf_1_70_Pc8_Pm8_seed1.pkl'
# filename = f'benchmark_CIFAR10/results_pkl/resnet8_backup_80_70_Pc8_Pm8_seed1.pkl'  

generation = 3
population = 5
conf = f'benchmark_CIFAR10/results_pkl/resnet32pareto_conf_{generation}_{population}_Pc8_Pm8_seed1.pkl'  
res = f'benchmark_CIFAR10/results_pkl/resnet32pareto_res_{generation}_{population}_Pc8_Pm8_seed1.pkl' 

# backup = f'benchmark_CIFAR10/results_pkl/resnet32_backup_80_70_Pc8_Pm8_seed1.pkl'  


with open(conf, 'rb') as file:
    conf_data = pickle.load(file)

with open(res, 'rb') as file:
    res_data = pickle.load(file)

print(conf_data)
print(res_data)


# with open(backup, 'rb') as file:
#     backup_data = pickle.load(file)

# print(backup_data)
# print(data.keys())
# print(f"designs: {data['designs']}")
# print(f"current_gen: {data['current_gen']}")
# print(f"Ng: {data['Ng']}")
# print(f"Np: {data['Np']}")
# print(f"Pc: {data['Pc']}")
# print(f"Pm: {data['Pm']}")
# print(f"net: {data['net']}")



