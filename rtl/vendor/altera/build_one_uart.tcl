# build_one_uart.tcl -- single-config UART demo build (called once per config).
# Reads globals: K_VAL MODE_VAL LABEL_VAL set via set_parameter in the caller.
# Usage:
#   quartus_sh -t vendor/altera/build_one_uart.tcl
# with env vars K_VAL / MODE_VAL / LABEL_VAL already in the environment,
# OR source this file after setting the three Tcl variables K_VAL MODE_VAL LABEL_VAL.
load_package flow

# Allow override via Tcl globals (set before sourcing) or fallback defaults.
if {![info exists K_VAL]}     { set K_VAL    0 }
if {![info exists MODE_VAL]}  { set MODE_VAL  0 }
if {![info exists LABEL_VAL]} { set LABEL_VAL "K0_trunc" }

set rtl_root "E:/axmac_rtl/rtl"
set pdir "$rtl_root/vendor/altera/build/uart_${LABEL_VAL}"
file mkdir $pdir
cd $pdir

puts "===== uart_${LABEL_VAL}  K=${K_VAL}  MODE=${MODE_VAL} ====="

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

set pins_tcl "$rtl_root/vendor/altera/mlp_top_ep4ce10_pins.tcl"
if {[file exists $pins_tcl]} {
    source $pins_tcl
} else {
    puts "WARNING: pins file not found"
}

execute_module -tool map
execute_module -tool fit
execute_module -tool asm
execute_module -tool sta

project_close

puts "Done: $pdir/demo.sof"
