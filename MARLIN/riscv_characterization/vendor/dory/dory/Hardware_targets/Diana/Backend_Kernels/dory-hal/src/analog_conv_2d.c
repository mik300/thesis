#include <kernels.h>
#include "utils.h"

extern L2_DATA uint32_t configuration_register[16];
extern L2_DATA Instruction_memory_object_analog instruction_memory_compiled_analog[2];

// Warning: we already assume here that A and B have the same shape!
int32_t analog_conv_2d(const void *input_L2, 
                const void *input,
                const void *weights_L2,
                const void *batchnorm_L2,
                const void *weights, 
                void *output,
                Layer_parameters * layer) {
    // STEP 1: definition and setting of the configuration registers
    // Sending the configuration register with the hardware functions.
    configuration_register[0] = 0x3000000;
    configuration_register[1] = 0x303001f;
    hwme_ana_memcpy_op((unsigned int) 0);
    hwme_ana_conf_addr_set((uint32_t) configuration_register);
    hwme_ana_conf_n_set(16);
    // STEP 2: definition and setting of the instruction memory
    //Instruction_memory_object *instruction_memory_compiled;
    analog_encoder_instruction_memory((uint32_t) input, (uint32_t) weights, (uint32_t) output, layer);
    hwme_ana_im_addr_set(instruction_memory_compiled_analog);
    hwme_ana_im_n_set(32); 
    hwme_ana_wt_addr_set((uint32_t) weights_L2);
    hwme_ana_simd_addr_set((uint32_t) batchnorm_L2); // TO SET
    hwme_ana_act_n_set(0); 
    hwme_ana_nb_tile_set(0);
    // STEP 3: triggering of the operations
    // The pointer to the instruction memory.
    hwme_ana_trigger_job();
    // Using the pulp specific function to diable the HWME accelerator.
    return 0;
}