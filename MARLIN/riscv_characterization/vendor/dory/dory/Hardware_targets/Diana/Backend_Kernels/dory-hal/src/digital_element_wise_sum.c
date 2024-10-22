#include <kernels.h>
#include "utils.h"

extern L2_DATA uint32_t configuration_register[16];
extern L2_DATA Instruction_memory_object_digital instruction_memory_compiled_digital[2];

// Warning: we already assume here that A and B have the same shape!
int32_t element_wise_sum(const void *input_L2, 
                        const void *inputA,
                        const void *weights_2_L2, 
                        const void *inputB, 
                        void *output,
                        Layer_parameters * layer) {
    layer->c = layer->cy * layer->cx * layer->c;
    Instruction_type mode = EWS;
    // STEP 1: definition and setting of the configuration registers
    // Sending the configuration register with the hardware functions.
    hwme_memcpy_op((unsigned int) 0);
    hwme_act_addr_set(input_L2);
    hwme_act_n_set(0);
    hwme_conf_addr_set((uint32_t) configuration_register);
    hwme_conf_n_set(16);

    // STEP 2: definition and setting of the instruction memory
    //Instruction_memory_object *instruction_memory_compiled;
    digital_encoder_instruction_memory(mode, (uint32_t) inputA, (uint32_t) inputB, (uint32_t) output, layer);
    hwme_im_addr_set(instruction_memory_compiled_digital);
    hwme_im_n_set(64);
    // STEP 3: triggering of the operations
    // The pointer to the instruction memory.
    hwme_trigger_job();
    // Using the pulp specific function to diable the HWME accelerator.
    return 0;
}
