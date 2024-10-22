#include <kernels.h>
#include "utils.h"

extern L2_DATA uint32_t configuration_register[16];
extern L2_DATA Instruction_memory_object_digital instruction_memory_compiled_digital[2];

// Warning: we already assume here that A and B have the same shape!
int32_t digital_conv_2d(const void *input_L2, 
                const void *input,
                const void *weights_L2,
                const void *weights, 
                void *output,
                Layer_parameters * layer) {
    Instruction_type mode = CONV;
    // STEP 1: definition and setting of the configuration registers
    // Sending the configuration register with the hardware functions.
    hwme_memcpy_op((unsigned int) 0);
    configuration_register[0] = 0x800;
    configuration_register[1] = 0xf;
    configuration_register[2] = 0x0;
    hwme_conf_addr_set((uint32_t) configuration_register);
    hwme_conf_n_set(16);
    // STEP 2: definition and setting of the instruction memory
    //Instruction_memory_object *instruction_memory_compiled;
    digital_encoder_instruction_memory(mode, (uint32_t) input, (uint32_t) weights, (uint32_t) output, layer);
    hwme_im_addr_set(instruction_memory_compiled_digital);
    hwme_im_n_set(64);
    hwme_act_addr_set(0); 
    hwme_act_n_set(0); 
    hwme_wt_conv_addr_set((uint32_t) weights_L2);
    hwme_wt_conv_n_set(( layer->c * layer->k * layer->fx * layer->fy + layer->k*4 ) /  4);
    // STEP 3: triggering of the operations
    // The pointer to the instruction memory.
    hwme_trigger_job();
    // Using the pulp specific function to diable the HWME accelerator.
    return 0;
}