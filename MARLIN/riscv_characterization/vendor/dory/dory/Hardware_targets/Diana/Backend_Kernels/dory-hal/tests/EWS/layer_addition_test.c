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
${verbose_log}

#include "${func_name}.h"
% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif

void ${func_name}(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l2_x      = (unsigned int)  real_arg[0];
  unsigned int l2_x2     = (unsigned int)  real_arg[1];
  unsigned int l2_y      = (unsigned int)  real_arg[2];
  unsigned int l2_W      = (unsigned int)  real_arg[3];
  volatile DMA_copy DMA_copy_x, DMA_copy_x2, DMA_copy_y;
  
  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = dory_dma_channel;

  DMA_copy_x2.hwc_to_chw = 0;
  DMA_copy_x2.stride_2d = ${x_stride_w_byte};
  DMA_copy_x2.stride_1d = ${x_stride_c_byte};
  DMA_copy_x2.dir = 1;
  DMA_copy_x2.dma_channel = dory_dma_channel;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${y_stride_w_byte};
  DMA_copy_y.stride_1d = ${y_stride_c_byte};
  DMA_copy_y.dir = 0;
  DMA_copy_y.dma_channel = dory_dma_channel;

  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + ${l1_x_offset}) + 0;
  DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x.length_1d_copy = ${x_tile_size_nif_byte};
  dory_dma_memcpy(DMA_copy_x);
  dory_accelerator_barrier();

  DMA_copy_x2.ext = l2_x2;
  DMA_copy_x2.loc = (l1_buffer + ${l1_x2_offset}) + 0;
  DMA_copy_x2.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x2.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x2.length_1d_copy = ${x_tile_size_nif_byte};
  dory_dma_memcpy(DMA_copy_x2);
  dory_accelerator_barrier();

  // tile loop indeces
  int _i_nof=0, _i_nif=0, _i_h=0, _i_w=0;

  // tile loop nest
  for(iter=0; iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}; iter++) {
    // loop nest is nof,h,w,(nif=0)
    _i_w += 1;
    if(_i_w==${tile_dim_w}) 
    {
      _i_w = 0;
      _i_h += 1;
      if(_i_h==${tile_dim_h}) 
      {
        _i_h = 0;
        _i_nif += 1;
        _i_nof += 1;
      }
    }

    if(iter<${tile_dim_nof}*${tile_dim_h}*${tile_dim_w}-1) 
    {
      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h, _i_w, _i_nif, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif},  ${conv_overlap1}, ${conv_overlap2},0, 0, 0, 0, ${x_data_size_byte});
      DMA_copy_x.loc = l1_buffer + ${l1_x_offset};
      DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
      DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
      DMA_copy_x.length_1d_copy = ${x_tile_size_nif};
      dory_dma_memcpy(DMA_copy_x);
      dory_accelerator_barrier();

      DMA_copy_x2.ext = dory_get_tile_3d(l2_x2, _i_h, _i_w, _i_nif, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif},  ${conv_overlap1}, ${conv_overlap2},0, 0, 0, 0, ${x_data_size_byte});
      DMA_copy_x2.loc = l1_buffer + ${l1_x2_offset};
      DMA_copy_x2.number_of_2d_copies = ${x_tile_size_h};
      DMA_copy_x2.number_of_1d_copies = ${x_tile_size_w};
      DMA_copy_x2.length_1d_copy = ${x_tile_size_nif};
      dory_dma_memcpy(DMA_copy_x2);
      dory_accelerator_barrier();
    }
    x  = l1_buffer + ${l1_x_offset};
    x2 = l1_buffer + ${l1_x2_offset};
    y  = l1_buffer + ${l1_y_offset};

    element_wise_sum(x, x2, y, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif});

    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h, _i_w, _i_nof, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${nof}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
    DMA_copy_y.loc = (l1_buffer + ${l1_y_offset}) + db_y;
    DMA_copy_y.number_of_2d_copies = ${y_tile_size_h};
    DMA_copy_y.number_of_1d_copies = ${y_tile_size_w};
    DMA_copy_y.length_1d_copy = ${y_tile_size_nof};
    dory_dma_memcpy(DMA_copy_y);
    dory_accelerator_barrier();
    
  }

}
