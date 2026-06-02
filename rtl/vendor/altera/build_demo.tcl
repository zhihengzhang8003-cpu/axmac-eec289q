# =============================================================================
# build_demo.tcl -- full Quartus flow for the on-chip demo (mlp_top_demo) on
# EP4CE10F17C8, producing a .sof to program the 野火征途 board.
#
# Pin assignments live in mlp_top_ep4ce10_pins.tcl (sourced if present). Until
# the real board pins are filled in there, this still runs map+fit with
# auto-placed pins so resource/timing can be checked -- but the generated .sof
# will NOT light the right LEDs until the pins are correct.
#
# Run from the rtl/ junction:
#   Set-Location E:\axmac_rtl
#   quartus_sh -t vendor\altera\build_demo.tcl
#
# Output: vendor/altera/build/demo/demo.sof
# =============================================================================

load_package flow

set rtl_root [pwd]
set pdir "$rtl_root/vendor/altera/build/demo"
file mkdir $pdir
cd $pdir

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

# K=2 / trunc demo; LEDs active-low for typical 野火 boards.
set_parameter -name K_PARAM 2
set_parameter -name MODE 0
set_parameter -name ACA_W 32
set_parameter -name LED_ACTIVE_LOW 1

# 50 MHz clock.
set sf [open "$pdir/demo.sdc" w]
puts $sf "create_clock -name clk -period 20.000 \[get_ports clk\]"
puts $sf "derive_clock_uncertainty"
close $sf
set_global_assignment -name SDC_FILE demo.sdc

# Real board pin assignments (clk / rst_n / led_*). Edit the pins file with
# values from the 野火征途 schematic before the .sof will work on hardware.
set pins_tcl "$rtl_root/vendor/altera/mlp_top_ep4ce10_pins.tcl"
if {[file exists $pins_tcl]} {
    puts "Sourcing board pins from $pins_tcl"
    source $pins_tcl
} else {
    puts "WARNING: $pins_tcl not found -- pins will be auto-placed (NOT board-correct)."
}

# Full flow to a programming file.
execute_module -tool map
execute_module -tool fit
execute_module -tool asm
execute_module -tool sta

puts ""
puts "Demo build done. Programming file: $pdir/demo.sof"
puts "Fit summary: $pdir/demo.fit.summary"
