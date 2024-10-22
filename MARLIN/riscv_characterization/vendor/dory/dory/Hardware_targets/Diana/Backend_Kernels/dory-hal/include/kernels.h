#include "utils.h"
#include "encoders_instruction_memory.h"

int32_t analog_conv_2d(const void *input_L2, 
                const void *input,
                const void *weights_L2,
                const void *batchnorm_L2,
                const void *weights, 
                void *output,
                Layer_parameters * layer);

int32_t digital_fully_connected(const void *input_L2, 
                const void *input,
                const void *weights_L2,
                const void *weights, 
                void *output,
                Layer_parameters * layer);

int32_t digital_conv_2d(const void *input_L2, 
                const void *input,
                const void *weights_L2,
                const void *weights, 
                void *output,
                Layer_parameters * layer);

int32_t digital_depthwise_conv_2d(const void *input_L2, 
                const void *input,
                const void *weights_L2,
                const void *weights, 
                void *output,
                Layer_parameters * layer);

int32_t digital_element_wise_sum(const void *input_L2, 
                        const void *inputA,
                        const void *weights_2_L2, 
                        const void *inputB, 
                        void *output,
                        Layer_parameters * layer);