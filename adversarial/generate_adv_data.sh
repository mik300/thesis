
#!/bin/bash

# Takes around 20 minutes to run

#adv. data generated by resnet8
python adversarial/generate_adv_data.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 108 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet8 --AT 1 --AT-epsilon 1 --AT-epochs 102 --execution-type transaxx
echo "3/24"

python adversarial/generate_adv_data.py --neural-network resnet8 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet8 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet8 --execution-type transaxx
echo "6/24"


#adv. data generated by resnet20

python adversarial/generate_adv_data.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 105 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet20 --AT 1 --AT-epsilon 1 --AT-epochs 117 --execution-type transaxx
echo "9/24"

python adversarial/generate_adv_data.py --neural-network resnet20 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet20 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet20 --execution-type transaxx
echo "12/24"

#adv. data generated by resnet32

python adversarial/generate_adv_data.py --neural-network resnet32 --AT 1 --AT-epsilon 1  --AT-epochs 105 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet32 --AT 1 --AT-epsilon 1 --AT-epochs 112 --execution-type transaxx
echo "15/24"

python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet32 --execution-type transaxx
echo "18/24"

#adv. data generated by resnet56

python adversarial/generate_adv_data.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 105 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet56 --AT 1 --AT-epsilon 1 --AT-epochs 106 --execution-type transaxx
echo "21/24"

python adversarial/generate_adv_data.py --neural-network resnet56 --execution-type float
python adversarial/generate_adv_data.py --neural-network resnet56 --execution-type quant
python adversarial/generate_adv_data.py --neural-network resnet56 --execution-type transaxx
echo "All adv. data has been generated"