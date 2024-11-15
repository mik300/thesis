from fractions import Fraction

attacks_dict = {
    'PGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'FGSM': {'eps': {'type': 'float', 'default': 8/255}},
    'BIM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'RFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'EOTPGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'eot_iter': {'type': 'int', 'default': 2}},
    'FFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 10/255}},
    'TPGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}},
    'MIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'decay': {'type': 'float', 'default': 1.0}, 'steps': {'type': 'int', 'default': 10}},
    'UPGD': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'random_start': {'type': 'bool', 'default': False}, 'loss': {'type': 'str', 'default': "ce"}, 'decay': {'type': 'float', 'default': 1.0}, 'eot_iter': {'type': 'int', 'default': 1}},
    'APGD': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'steps': {'type': 'int', 'default': 10}, 'n_restarts': {'type': 'int', 'default': 1}, 'seed': {'type': 'int', 'default': 0}, 'loss': {'type': 'str', 'default': "ce"}, 'eot_iter': {'type': 'int', 'default': 1}, 'rho': {'type': 'float', 'default': 0.75}, 'verbose': {'type': 'bool', 'default': False}},
    'APGDT': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'steps': {'type': 'int', 'default': 10}, 'n_restarts': {'type': 'int', 'default': 1}, 'seed': {'type': 'int', 'default': 0}, 'eot_iter': {'type': 'int', 'default': 1}, 'rho': {'type': 'float', 'default': 0.75}, 'verbose': {'type': 'bool', 'default': False}, 'n_classes': {'type': 'int', 'default': 10}},
    'DIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'decay': {'type': 'float', 'default': 0.0}, 'steps': {'type': 'int', 'default': 10}, 'resize_rate': {'type': 'float', 'default': 0.9}, 'diversity_prob': {'type': 'float', 'default': 0.5}, 'random_start': {'type': 'bool', 'default': False}},
    'TIFGSM':  {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'decay': {'type': 'float', 'default': 1.0}, 'kernel_name': {'type': 'str', 'default': "gaussian"}, 'len_kernel': {'type': 'int', 'default': 15}, 'nsig': {'type': 'int', 'default': 3}, 'resize_rate': {'type': 'float', 'default': 0.9}, 'diversity_prob': {'type': 'float', 'default': 0.5}, 'random_start': {'type': 'bool', 'default': False}},
    'Jitter': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'random_start': {'type': 'bool', 'default': True}},
    'NIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'decay': {'type': 'float', 'default': 1.0}, 'steps': {'type': 'int', 'default': 10}},
    'PGDRS': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'noise_type': {'type': 'str', 'default': "guassian"}, 'noise_sd': {'type': 'float', 'default': 0.5}, 'noise_batch_size': {'type': 'int', 'default': 5}, 'batch_max': {'type': 'int', 'default': 2048}},
    'SINIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'decay': {'type': 'float', 'default': 1.0}, 'm': {'type': 'int', 'default': 5}},
    'VMIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'decay': {'type': 'float', 'default': 1.0}, 'N': {'type': 'int', 'default': 5}, 'beta': {'type': 'float', 'default': 3/2}},
    'VNIFGSM': {'eps': {'type': 'float', 'default': 8/255}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'decay': {'type': 'float', 'default': 1.0}, 'N': {'type': 'int', 'default': 5}, 'beta': {'type': 'float', 'default': 3/2}},
    'SPSA': {'eps': {'type': 'float', 'default': 8/255}, 'delta': {'type': 'float', 'default': 0.01}, 'lr': {'type': 'float', 'default': 0.01}, 'nb_iter': {'type': 'int', 'default': 1}, 'nb_sample': {'type': 'int', 'default': 128}, 'max_batch_size': {'type': 'int', 'default': 64}},
    'JSMA': {'theta': {'type': 'float', 'default': 1.0}, 'gamma': {'type': 'float', 'default': 0.1}},
    'EADL1': {'kappa': {'type': 'float', 'default': 0.0}, 'lr': {'type': 'float', 'default': 0.01}, 'binary_search_steps': {'type': 'int', 'default': 9}, 'max_iterations': {'type': 'int', 'default': 100}, 'abort_early': {'type': 'bool', 'default': True}, 'initial_const': {'type': 'float', 'default': 0.001}, 'beta': {'type': 'float', 'default': 0.001}}, 
    'EADEN': {'kappa': {'type': 'float', 'default': 0.0}, 'lr': {'type': 'float', 'default': 0.01}, 'binary_search_steps': {'type': 'int', 'default': 9}, 'max_iterations': {'type': 'int', 'default': 100}, 'abort_early': {'type': 'bool', 'default': True}, 'initial_const': {'type': 'float', 'default': 0.001}, 'beta': {'type': 'float', 'default': 0.001}}, 
    'PIFGSM': {'max_epsilon': {'type': 'float', 'default': 16/255}, 'num_iter_set': {'type': 'float', 'default': 10.0}, 'momentum': {'type': 'float', 'default': 1.0}, 'amplification': {'type': 'float', 'default': 10.0}, 'prob': {'type': 'float', 'default': 0.7}},
    'PIFGSMPP': {'max_epsilon': {'type': 'float', 'default': 16/255}, 'num_iter_set': {'type': 'float', 'default': 10.0}, 'momentum': {'type': 'float', 'default': 1.0}, 'amplification': {'type': 'float', 'default': 10.0}, 'prob': {'type': 'float', 'default': 0.7}, 'project_factor': {'type': 'float', 'default': 0.8}},
    'CW': {'c': {'type': 'float', 'default': 1.0}, 'kappa': {'type': 'float', 'default': 0.0}, 'lr': {'type': 'float', 'default': 0.01}},
    'PGDL2': {'eps': {'type': 'float', 'default': 1.0}, 'alpha': {'type': 'float', 'default': 2/255}, 'steps': {'type': 'int', 'default': 10}, 'random_start': {'type': 'bool', 'default': True}, 'eps_for_division': {'type': 'float', 'default': 1e-10}},
    'DeepFool': {'steps': {'type': 'int', 'default': 50}, 'overshoot': {'type': 'float', 'default': 0.02}},
    'PGDRSL2': {'eps': {'type': 'float', 'default': 1.0}, 'alpha': {'type': 'float', 'default': 0.2}, 'steps': {'type': 'int', 'default': 10}, 'noise_type': {'type': 'str', 'default': "guassian"}, 'noise_sd': {'type': 'float', 'default': 0.5}, 'noise_batch_size': {'type': 'int', 'default': 5}, 'batch_max': {'type': 'int', 'default': 2048}, 'random_start': {'type': 'bool', 'default': True}},
    'SparseFool': {'steps': {'type': 'int', 'default': 10}, 'lam': {'type': 'float', 'default': 3.0}, 'overshoot': {'type': 'float', 'default': 0.02}},
    'OnePixel': {'pixels': {'type': 'int', 'default': 1}, 'steps': {'type': 'int', 'default': 10}, 'popsize': {'type': 'int', 'default': 10}, 'inf_batch': {'type': 'int', 'default': 128}},
    'Pixle': {'x_dimensions': {'type': 'tuple', 'default': (2, 10)}, 'y_dimensions': {'type': 'tuple', 'default': (2, 10)}, 'pixel_mapping': {'type': 'str', 'default': "random"}, 'restarts': {'type': 'int', 'default': 20}, 'max_iterations': {'type': 'int', 'default': 10}, 'update_each_iteration': {'type': 'bool', 'default': False}},
    'FAB': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'steps': {'type': 'int', 'default': 10}, 'n_restarts': {'type': 'int', 'default': 1}, 'alpha_max': {'type': 'float', 'default': 0.1}, 'eta': {'type': 'float', 'default': 1.05}, 'beta': {'type': 'float', 'default': 0.9}, 'verbose': {'type': 'bool', 'default': False}, 'seed': {'type': 'int', 'default': 0}, 'multi_targeted': {'type': 'bool', 'default': False}, 'n_classes': {'type': 'int', 'default': 10}},
    'AutoAttack': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'version': {'type': 'str', 'default': "standard"}, 'n_classes': {'type': 'int', 'default': 10}, 'seed': {'type': 'int', 'default': 0}, 'verbose': {'type': 'bool', 'default': False}},
    'Square': {'norm': {'type': 'str', 'default': "Linf"}, 'eps': {'type': 'float', 'default': 8/255}, 'n_queries': {'type': 'int', 'default': 5000}, 'n_restarts': {'type': 'int', 'default': 1}, 'p_init': {'type': 'float', 'default': 0.8}, 'loss': {'type': 'str', 'default': "margin"}, 'resc_schedule': {'type': 'bool', 'default': True}, 'seed': {'type': 'int', 'default': 0}, 'verbose': {'type': 'bool', 'default': False}, 'targeted': {'type': 'bool', 'default': False}}, 
}


def get_attack():
    default_attack_type = "PGD"
    attack_type = input(f"Enter the attack type [default: {default_attack_type}]: ") or default_attack_type
    if attack_type in attacks_dict:
        params = {}
        for param, info in attacks_dict[attack_type].items():
            default_value = info.get('default')
            param_type = info['type']
            value_str = input(f"Enter value for {param} ({param_type}) : ")
            
            # Parse input based on specified type
            if value_str == "":
                value = default_value
            else:
                if param_type == 'int':
                    value = int(value_str)
                elif param_type == 'float':
                    try:
                        # If that fails, try parsing as a float
                        value = float(value_str)
                    except ValueError:
                        # If both int and float parsing fail, treat it as a fraction
                        value = float(Fraction(value_str))
                elif isinstance(param_type, str):
                    value = str(value_str)
                else:
                    raise ValueError(f"Unsupported parameter type '{param_type}' for {param}")

            params[param] = value
        return attack_type, params
    else:
        raise ValueError(f"Attack type '{attack_type}' not recognized.")