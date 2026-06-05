# =============================================================================
# build_uart_demo.tcl -- build 5 UART-enabled demo configs for on-chip testing.
#
# Configs: K=0/2/4/6 trunc, K=4 round.
# Each → vendor/altera/build/uart_K<K>_<mode>/demo.sof
#
# Run from E:\axmac_rtl:
#   quartus_sh -t vendor\altera\build_uart_demo.tcl
# =============================================================================
load_package flow

set rtl_root [pwd]
set pins_tcl "$rtl_root/vendor/altera/mlp_top_ep4ce10_pins.tcl"

# {K MODE label}
set configs {
    {0 0 K0_trunc}
    {2 0 K2_trunc}
    {4 0 K4_trunc}
    {6 0 K6_trunc}
    {4 1 K4_round}
}

foreach cfg $configs {
    set K     [lindex $cfg 0]
    set MODE  [lindex $cfg 1]
    set label [lindex $cfg 2]

    set pdir "$rtl_root/vendor/altera/build/uart_${label}"
    file mkdir $pdir
    cd $pdir

    puts ""
    puts "===== Building uart_${label} (K=${K} MODE=${MODE}) ====="

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

    set_parameter -name K_PARAM       $K
    set_parameter -name MODE          $MODE
    set_parameter -name ACA_W         32
    set_parameter -name LED_ACTIVE_LOW 1

    set sf [open "$pdir/demo.sdc" w]
    puts $sf "create_clock -name clk -period 20.000 \[get_ports clk\]"
    puts $sf "derive_clock_uncertainty"
    close $sf
    set_global_assignment -name SDC_FILE demo.sdc

    if {[file exists $pins_tcl]} {
        source $pins_tcl
    } else {
        puts "WARNING: pins file not found -- pins auto-placed."
    }

    execute_module -tool map
    execute_module -tool fit
    execute_module -tool asm
    execute_module -tool sta

    puts "Done: $pdir/demo.sof"
    project_close
}

puts ""
puts "All configs built. SOF files:"
foreach cfg $configs {
    set label [lindex $cfg 2]
    puts "  vendor/altera/build/uart_${label}/demo.sof"
}
