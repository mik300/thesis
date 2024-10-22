/*
 * add_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include "element_wise_sum.h"
#define H 1
#define W 7
#define C 1


L2_DATA uint8_t x_L2[(H*W*C)];
L2_DATA uint8_t x2_L2[(H*W*C)];
L2_DATA uint8_t y_L2[padding(H*W*C, DIG_HWPE_PARALLELISM)];

void main() 
{

    uint32_t x = 0x0;
    uint32_t x2 = 0x80;
    uint32_t y = 0x100;

    for (int i = 0; i < (H*W*C); i++)
    {
        x_L2[i] = i*2+2;
        x2_L2[i] = i*2+1; 
        y_L2[i] = 1; 
    }
    Layer_parameters layer;
    layer.cx = 1;
    layer.cy = 1;
    layer.c  = H*W*C;
    
    plp_hwme_enable();
    memcpy_dig((unsigned int*) x_L2, x, H*W*C, 0, 1);
    global_sync();
    memcpy_dig((unsigned int*) x2_L2, x2, H*W*C, 0, 1);
    global_sync();
    element_wise_sum(x, x2, y, layer);
    global_sync();
    memcpy_dig((unsigned int*) y_L2, y, H*W*C, 1, 1);
    plp_hwme_disable();
    for (int i = 0; i < (H*W*C); i++)
    {
        printf("Element %d: %d + %d = %d\n", i, x_L2[i], x2_L2[i], y_L2[i]);
    }
    
  

}
