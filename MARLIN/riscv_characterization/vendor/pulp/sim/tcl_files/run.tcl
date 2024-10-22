#!/bin/bash
set SYN_LIB "" ;# set this variable with the path to the library used for synthesis
set TB "vopt_tb -L models_lib -L vip_lib -L $SYN_LIB"
source ./tcl_files/config/vsim.tcl
