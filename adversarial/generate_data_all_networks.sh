
#!/bin/bash

# This script takes around 7h to run (PGD)
# 7:41h (FGSM)

#resnet8 against all other networks
python adversarial/generate_data_for_plots.py --neural-network resnet8 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet8 --adv-AT 1
echo "3/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet8 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet20 --adv-AT 1
echo "6/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet8 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet32 --adv-AT 1
echo "9/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet8 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --AT-epochs-float 108 --adv-neural-network resnet56 --adv-AT 1
echo "12/48" > adversarial/log/generate_data_all_networks.log

#resnet20 against all other networks
python adversarial/generate_data_for_plots.py --neural-network resnet20 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet8 --adv-AT 1
echo "15/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet20 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet20 --adv-AT 1
echo "18/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet20 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet32 --adv-AT 1
echo "21/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet20 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --AT-epochs-float 105 --adv-neural-network resnet56 --adv-AT 1
echo "24/48" > adversarial/log/generate_data_all_networks.log

#resnet32 against all other networks
python adversarial/generate_data_for_plots.py --neural-network resnet32 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet8 --adv-AT 1
echo "27/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet32 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet20 --adv-AT 1
echo "30/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet32 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet32 --adv-AT 1
echo "33/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet32 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --AT-epochs-float 105 --adv-neural-network resnet56 --adv-AT 1
echo "36/48" > adversarial/log/generate_data_all_networks.log

#resnet56 against all other networks
python adversarial/generate_data_for_plots.py --neural-network resnet56 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet8 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet8 --adv-AT 1
echo "39/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet56 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet20 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet20 --adv-AT 1
echo "42/48" > adversarial/log/generate_data_all_networks.log

python adversarial/generate_data_for_plots.py --neural-network resnet56 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet32 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet32 --adv-AT 1
echo "45/48" > adversarial/log/generate_data_all_networks.log 

python adversarial/generate_data_for_plots.py --neural-network resnet56 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet56 --adv-AT 0
python adversarial/generate_data_for_plots.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --AT-epochs-float 105 --adv-neural-network resnet56 --adv-AT 1
echo "All data needed for plots is generated"


