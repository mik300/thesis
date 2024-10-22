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
#define H 16
#define W 16
#define C 16
#define K 16
#define Fx 1
#define Fy 1


L2_DATA uint8_t x_L2[(H*W*C)];
L2_DATA uint8_t W_L2[(K*C*Fx*Fy + K)];
L2_DATA uint8_t y_L2[(H*W*K)];

void main() 
{

    uint32_t x = 0x0;
    uint32_t y = 0x20000;

    for (int i = 0; i < (H*W*C); i++)
    {
        x_L2[i] = i%32;
    }
    for (int i = 0; i < (K*C*Fx*Fy + K); i++)
    {
        W_L2[i] = 1;
    }
    Layer_parameters layer;
    layer.c = C;
    layer.k = K;
    layer.cx = H;
    layer.cy = W;
    layer.fx = Fx;
    layer.fy = Fy;
    layer.ox = H;
    layer.oy = W;
    layer.activation_function = 0;
    layer.dilation = 1;
    
    plp_hwme_enable();
    memcpy_dig((unsigned int*) x_L2, x, (H*W*C), 0, 1);
    global_sync();
    conv_2d(x, W_L2, y, layer);
    global_sync();
    memcpy_dig((unsigned int*) y_L2, y, (H*W*K), 1, 1);
    plp_hwme_disable();
    for (int i = 0; i < (10); i++)
    {
        printf("Element %d: %d + %d = %d\n", i, x_L2[i], W_L2[i], y_L2[i]);
    }
    
  

}
