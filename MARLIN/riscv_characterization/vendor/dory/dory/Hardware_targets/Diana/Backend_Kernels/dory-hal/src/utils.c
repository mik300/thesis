#include "utils.h"

L2_DATA uint32_t configuration_register[16] = {0};
L2_DATA Instruction_memory_object_analog instruction_memory_compiled_analog[2] = {0};
L2_DATA Instruction_memory_object_digital instruction_memory_compiled_digital[2] = {0};

L2_DATA __attribute__((aligned(16))) uint32_t ima_boot[] = {
    0x3f,
    0x8000000,
    0x0,
    0x400000,
    0x20200,
    0x70002,
    0x20011,
    0x10002,
    0x1,
    0x20050001,
    0x100ffff,
    0xf000,
    0x9000084,
    0x0,
    0x0,
    0x0,
    0x3f,
    0x0,
    0x0,
    0x0,
    0x0,
    0x70000,
    0x0,
    0x10000,
    0x1,
    0x50001,
    0x10a0001,
    0xf001,
    0x84,
    0x0,
    0x0,
    0x0,
};

L2_DATA __attribute__((aligned(16))) uint32_t cra_boot[] = { 
0x1000000,// cfg.end_time[15:8] cfg.unit_time[7:0] //  cfg.use_blockcolumn[0] // 
0x01010001,// cfg.dac_timing[15:8] cfg.nb_dac_bits[7:0] //  cfg.nb_act_bits[7:0]  // 
0x0,// cfg.csbias_mirror[15:8] cfg.csbias_source[7:0] //  cfg.csbias_observe[2] cfg.csbias_force[1] cfg.force_activation[0] // 
0x0,// dbg.pd_col_c[15:0] //  dbg.pd_col_b[11:4] dbg.pd_col_a[3:0] // 
0x0,// dbg.sw_reset[0] //  dbg.col_connect_min[1] dbg.col_connect_plus[0]  // 
0x0,// empty field //  stp_loop_instr[10:6] str_loop_instr[5:1] inf_loop[0] // 
0x0,// empty field //  empty field // 
0x0,// empty field //  empty field //
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0 
};

void global_sync_digital(){
  while (hwme_get_status() != 0);
}

void global_sync_analog(){
  while (hwme_ana_get_status() != 0);
}

uint32_t* pad_out = (uint32_t*)0x1a10100c;
uint32_t* pad_in  = (uint32_t*)0x1a101008;


static inline void init_boot_gpio(){
    //INIT GPIO
    int i,j;
    for(i = 0; i < 32; i = i + 1) {
        rt_pad_set_function(i, 1);
        rt_gpio_init(0, i);
        if (i==18)
            rt_gpio_set_dir(0, 1 << i, RT_GPIO_IS_IN);
        else
            rt_gpio_set_dir(0, 1 << i, RT_GPIO_IS_OUT);
    }
    //WAIT
    j=0;
    while (j < 25000) {
        j++;
    }
}

static inline void set_pin(int pin){
    uint32_t pval = 0;
    pval = *pad_out;
    *pad_out = (uint32_t) 1 << pin | pval;
}

static inline void reset_pin(int pin){
    uint32_t pval = 0;
    pval = *pad_out;
    *pad_out = (uint32_t) ~(1 << pin) & pval;
}

static inline void wait_pin(int pin, int pol){
    int j=0;
    int mask = 1 << pin;
    while((uint32_t)(*pad_in & mask) >> pin == pol){
        j++;
    }
}

void boot_diana(){

  //----------------SIZING THE .H DATA--------------------------------
  volatile int size_array_cr_ania=sizeof(cra_boot)/sizeof(cra_boot[0]);

  //----------BOOT PROCEDURE------------------------
  init_boot_gpio();
  set_pin(17);


  hwme_ana_memcpy_op((unsigned int) 0);
  // set up addresses
  hwme_ana_conf_addr_set(&cra_boot[0]);
  hwme_ana_conf_n_set(size_array_cr_ania);
  hwme_ana_im_addr_set( &ima_boot[0]);
  unsigned int imSizeIn32Bit;
  imSizeIn32Bit = 2*16;
  hwme_ana_im_n_set(imSizeIn32Bit);
  hwme_ana_wt_addr_set(0);
  hwme_ana_simd_addr_set(0);
  hwme_ana_act_n_set(0);
  hwme_ana_nb_tile_set(0);
  hwme_ana_trigger_job();
  global_sync_analog();
  reset_pin(17);
  wait_pin(18,0);
}