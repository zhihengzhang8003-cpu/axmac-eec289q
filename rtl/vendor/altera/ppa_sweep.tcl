# =============================================================================
# ppa_sweep.tcl -- area + power sweep of mlp_top across rounding modes and ACA
# windows, for paper Contribution A's hardware comparison.
#
# For each design point it runs the full Quartus flow (Analysis & Synthesis ->
# Fit -> STA -> PowerPlay) on mlp_top targeting EP4CE10F17C8, with virtual pins
# on the wide result bus and a 50 MHz clock, then scrapes LE / registers / DSP
# and PowerPlay thermal power into vendor/altera/build/ppa_results.csv.
#
# Design points:
#   trunc_K6 / round_K6 / stoch_K6  -- rounding-mode cost at K=6, W=32 (exact
#                                      adder), isolates the trunc vs round vs
#                                      LFSR-stochastic hardware gap.
#   acaW32_K0 / acaW8_K0 / acaW4_K0 -- ACA approximate-adder area as the carry
#                                      window shrinks, at K=0 (no truncation).
#
# Run from the rtl/ junction:
#   Set-Location E:\axmac_rtl
#   quartus_sh -t vendor\altera\ppa_sweep.tcl
# =============================================================================

load_package flow
load_package report

set rtl_root    [pwd]
set results_csv "$rtl_root/vendor/altera/build/ppa_results.csv"

# {label MODE K_PARAM ACA_W}
set configs [list \
    [list trunc_K6  0 6 32] \
    [list round_K6  1 6 32] \
    [list stoch_K6  2 6 32] \
    [list acaW32_K0 0 0 32] \
    [list acaW8_K0  0 0 8]  \
    [list acaW4_K0  0 0 4]  \
]

file mkdir "$rtl_root/vendor/altera/build"
set rf [open $results_csv w]
puts $rf "label,mode,K,aca_w,logic_elements,registers,dsp9,total_thermal_mw,core_dynamic_mw"
close $rf

# Pull "Key : value" from a Quartus *.summary text file (exact prefix match
# on the trimmed line). More robust than the report-panel API across versions.
proc summary_field {file key} {
    if {![file exists $file]} { return "NA" }
    set fh [open $file r]
    set val "NA"
    while {[gets $fh line] >= 0} {
        set line [string trim $line]
        if {[string first $key $line] == 0} {
            set idx [string first ":" $line]
            if {$idx >= 0} {
                set val [string trim [string range $line [expr {$idx + 1}] end]]
            }
            break
        }
    }
    close $fh
    return $val
}

# First numeric token of a value like "1,685 / 10,320 ( 16 % )" -> "1685".
proc first_num {s} {
    set s [string map {"," ""} $s]
    return [lindex [split $s] 0]
}

foreach cfg $configs {
    set label [lindex $cfg 0]
    set mode  [lindex $cfg 1]
    set kk    [lindex $cfg 2]
    set ww    [lindex $cfg 3]

    puts "=================================================="
    puts " PPA config: $label   MODE=$mode K=$kk ACA_W=$ww"
    puts "=================================================="

    set pdir "$rtl_root/vendor/altera/build/ppa_$label"
    file mkdir $pdir
    cd $pdir

    project_new ppa -overwrite

    set_global_assignment -name FAMILY "Cyclone IV E"
    set_global_assignment -name DEVICE EP4CE10F17C8
    set_global_assignment -name TOP_LEVEL_ENTITY mlp_top

    # Sources (relative to this project dir = vendor/altera/build/ppa_<label>).
    set_global_assignment -name VERILOG_FILE ../../../../src/mac_unit.v
    set_global_assignment -name VERILOG_FILE ../../../../src/aca_adder.v
    set_global_assignment -name VERILOG_FILE ../../../../src/mac_array.v
    set_global_assignment -name VERILOG_FILE ../../../../src/lfsr.v
    set_global_assignment -name VERILOG_FILE ../../../../src/mlp_top.v

    # Design-point parameters.
    set_parameter -name MODE     $mode
    set_parameter -name K_PARAM  $kk
    set_parameter -name ACA_W    $ww
    # Memory init images (keep the weight ROMs from being optimized away).
    set_parameter -name X_FILE   "../../../../golden/mlp_toy/x.mem"
    set_parameter -name W0_FILE  "../../../../golden/mlp_toy/w0.mem"
    set_parameter -name B0_FILE  "../../../../golden/mlp_toy/b0.mem"
    set_parameter -name W1_FILE  "../../../../golden/mlp_toy/w1.mem"
    set_parameter -name B1_FILE  "../../../../golden/mlp_toy/b1.mem"

    # Wide result bus -> virtual pins (EP4CE10 has only 179 real I/O).
    set_instance_assignment -name VIRTUAL_PIN ON -to "result_flat\[*\]"

    # 50 MHz clock for STA + PowerPlay.
    set sf [open "$pdir/clk.sdc" w]
    puts $sf "create_clock -name clk -period 20.000 \[get_ports clk\]"
    puts $sf "derive_clock_uncertainty"
    close $sf
    set_global_assignment -name SDC_FILE clk.sdc

    set_global_assignment -name POWER_DEFAULT_INPUT_IO_TOGGLE_RATE "12.5%"
    set_global_assignment -name POWER_USE_INPUT_FILES OFF

    # Full flow. The Assembler (asm) must run before PowerPlay -- pow reads
    # the assembled netlist.
    execute_module -tool map
    execute_module -tool fit
    execute_module -tool asm
    execute_module -tool sta
    execute_module -tool pow

    # Scrape area from the Fitter summary, power from the PowerPlay summary
    # (both are "key : value" text files in this project dir).
    set fit_sum "ppa.fit.summary"
    set pow_sum "ppa.pow.summary"
    set le   [first_num [summary_field $fit_sum "Total logic elements"]]
    set regs [first_num [summary_field $fit_sum "Total registers"]]
    set dsp  [first_num [summary_field $fit_sum "Embedded Multiplier 9-bit elements"]]
    set tot_pwr  [first_num [summary_field $pow_sum "Total Thermal Power Dissipation"]]
    set core_pwr [first_num [summary_field $pow_sum "Core Dynamic Thermal Power Dissipation"]]

    set rf [open $results_csv a]
    puts $rf "$label,$mode,$kk,$ww,$le,$regs,$dsp,$tot_pwr,$core_pwr"
    close $rf

    puts "  -> LE=$le  regs=$regs  DSP=$dsp  Ptot=$tot_pwr  Pcore=$core_pwr"

    project_close
    cd $rtl_root
}

puts ""
puts "=================================================="
puts " PPA sweep done. Results: $results_csv"
puts "=================================================="

# =============================================================================
# DRUM-4 standalone synthesis
# Synthesises drum_multiplier.v in isolation to obtain LE / timing numbers
# directly comparable to the K=4 trunc/round entries above.
#   project name : ppa_drum_k4
#   top          : drum_multiplier
#   output dir   : E:/uart_builds/ppa_drum_k4   (matches on-chip build paths)
# =============================================================================

puts ""
puts "=================================================="
puts " DRUM-4 standalone PPA"
puts "=================================================="

set drum_dir "E:/uart_builds/ppa_drum_k4"
file mkdir $drum_dir
cd $drum_dir

project_new ppa_drum_k4 -overwrite

set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE10F17C8
set_global_assignment -name TOP_LEVEL_ENTITY drum_multiplier

# Source files (relative to the new project dir under E:/uart_builds/).
# drum_multiplier.v is the sole RTL file required for standalone synthesis.
set_global_assignment -name VERILOG_FILE "$rtl_root/src/drum_multiplier.v"

# Parameters: N_BITS=8, K_DRUM=4 (INT8 DRUM-4 as used in the paper).
set_parameter -name N_BITS 8
set_parameter -name K_DRUM 4

# Virtual pins on all output ports so the fitter does not optimise them away.
set_instance_assignment -name VIRTUAL_PIN ON -to "product\[*\]"

# 50 MHz timing constraint (consistent with all other PPA configs).
set sf [open "$drum_dir/clk_drum.sdc" w]
puts $sf "# No clock ports on drum_multiplier (pure combinational)."
puts $sf "# Constrain the primary output for STA coverage."
puts $sf "create_clock -name virt_clk -period 20.000"
puts $sf "set_output_delay -clock virt_clk -max 0 \[get_ports product\[*\]\]"
close $sf
set_global_assignment -name SDC_FILE clk_drum.sdc

set_global_assignment -name POWER_DEFAULT_INPUT_IO_TOGGLE_RATE "12.5%"
set_global_assignment -name POWER_USE_INPUT_FILES OFF

execute_module -tool map
execute_module -tool fit
execute_module -tool asm
execute_module -tool sta
execute_module -tool pow

set fit_sum "ppa_drum_k4.fit.summary"
set pow_sum "ppa_drum_k4.pow.summary"
set le   [first_num [summary_field $fit_sum "Total logic elements"]]
set regs [first_num [summary_field $fit_sum "Total registers"]]
set dsp  [first_num [summary_field $fit_sum "Embedded Multiplier 9-bit elements"]]
set tot_pwr  [first_num [summary_field $pow_sum "Total Thermal Power Dissipation"]]
set core_pwr [first_num [summary_field $pow_sum "Core Dynamic Thermal Power Dissipation"]]

# Append result to the shared ppa_results.csv using the same column schema.
set rf [open $results_csv a]
puts $rf "drum_K4,NA,4,NA,$le,$regs,$dsp,$tot_pwr,$core_pwr"
close $rf

puts "  -> LE=$le  regs=$regs  DSP=$dsp  Ptot=$tot_pwr  Pcore=$core_pwr"

project_close
cd $rtl_root

puts ""
puts "=================================================="
puts " DRUM-4 synthesis done. Results appended to: $results_csv"
puts "=================================================="
