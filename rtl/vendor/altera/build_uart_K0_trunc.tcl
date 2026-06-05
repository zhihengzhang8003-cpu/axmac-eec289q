load_package flow

set rtl_root "E:/axmac_rtl/rtl"
set K_VAL    0
set MODE_VAL 0
set LABEL    "K0_trunc"
set pdir     "$rtl_root/vendor/altera/build/uart_${LABEL}"

file mkdir $pdir
cd $pdir

puts "===== uart_${LABEL}  K=${K_VAL}  MODE=${MODE_VAL}  pwd=[pwd] ====="

project_new demo -overwrite

set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE10F17C8
set_global_assignment -name TOP_LEVEL_ENTITY mlp_top_demo

set_global_assignment -name VERILOG_FILE ../../../../src/mac_unit.v
set_global_assignment -name VERILOG_FILE ../../../../src/aca_adder.v
set_global_assignment -name VERILOG_FILE ../../../../src/mac_array.v
set_global_assignment -name VERILOG_FILE ../../../../src/lfsr.v
set_global_assignment -name VERILOG_FILE ../../../../src/mlp_top.v
set_global_assignment -name VERILOG_FILE ../../../../src/uart_tx.v
set_global_assignment -name VERILOG_FILE ../../../../src/uart_framer.v
set_global_assignment -name VERILOG_FILE ../../../../src/mlp_top_demo.v

set_parameter -name K_PARAM        $K_VAL
set_parameter -name MODE           $MODE_VAL
set_parameter -name ACA_W          32
set_parameter -name LED_ACTIVE_LOW 1

set sf [open "$pdir/demo.sdc" w]
puts $sf "create_clock -name clk -period 20.000 \[get_ports clk\]"
puts $sf "derive_clock_uncertainty"
close $sf
set_global_assignment -name SDC_FILE demo.sdc

source "$rtl_root/vendor/altera/mlp_top_ep4ce10_pins.tcl"

execute_module -tool map
execute_module -tool fit
execute_module -tool asm
execute_module -tool sta

project_close
puts "DONE: $pdir/demo.sof"
