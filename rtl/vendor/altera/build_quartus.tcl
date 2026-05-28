# =============================================================================
# build_quartus.tcl -- Analysis & Synthesis of mac_array (R=4, C=4) for
# Cyclone IV E EP4CE10F17C8 (野火征途). Reports resource usage so we know
# whether the design fits before committing to a Fit + pin-assignment flow.
#
# Why "Analysis & Synthesis only":
#   mac_array's R*C*P_BITS = 512-wide acc_flat / rnd_flat ports total ~1100
#   bits of top-level I/O -- EP4CE10F17C8 has only 179 user pins. A Fit pass
#   would fail on pinout. For Phase 5b-pre we only care about the area
#   estimate (Logic Elements, embedded multipliers, memory bits). A real
#   demo top-level wrapping mac_array behind a UART/AXI-Stream comes in
#   Phase 4 + Phase 5b-full.
#
# Run (from inside the rtl/ junction):
#   Set-Location E:\axmac_rtl
#   quartus_sh -t vendor\altera\build_quartus.tcl
#
# Output:
#   vendor/altera/build/mac_array.map.rpt       (full Quartus report)
#   vendor/altera/build/mac_array.map.summary
#   console: resource usage summary table
# =============================================================================

load_package flow

# -----------------------------------------------------------------------------
# Project layout: keep all Quartus droppings under vendor/altera/build/ so
# they don't pollute src/ or tb/.
# -----------------------------------------------------------------------------
set rtl_root [pwd]
set proj_dir [file join $rtl_root vendor altera build]
file mkdir $proj_dir
cd $proj_dir

project_new mac_array -overwrite

# -----------------------------------------------------------------------------
# Device. 野火征途 uses EP4CE10F17C8 (Cyclone IV E, 256-pin FBGA, speed -8).
# Change to EP4CE10E22C8 / EP4CE10F17I7 etc. if your specific board differs.
# -----------------------------------------------------------------------------
set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE EP4CE10F17C8

# -----------------------------------------------------------------------------
# Sources (paths relative to $proj_dir = vendor/altera/build).
# Order matters: leaf modules first.
# -----------------------------------------------------------------------------
set_global_assignment -name VERILOG_FILE ../../../src/mac_unit.v
set_global_assignment -name VERILOG_FILE ../../../src/aca_adder.v
set_global_assignment -name VERILOG_FILE ../../../src/mac_array.v

set_global_assignment -name TOP_LEVEL_ENTITY mac_array

# (No SDC at the area-estimate stage -- STA runs after the Phase 4 top-level
# wrapper gets real clock/pin constraints. Quartus emits a harmless "no SDC"
# warning here; ignore it.)

# -----------------------------------------------------------------------------
# Run Analysis & Synthesis (Quartus "Map" stage). No Fit, no PowerPlay yet.
# -----------------------------------------------------------------------------
puts "=========================================="
puts "Running Quartus Analysis & Synthesis ..."
puts "=========================================="
execute_module -tool map

# -----------------------------------------------------------------------------
# Dump the resource-usage panel to the console.
# -----------------------------------------------------------------------------
load_report

proc dump_panel {name} {
    set pid [get_report_panel_id $name]
    if {$pid < 0} {
        puts "  (panel '$name' not found)"
        return
    }
    set nrow [get_number_of_rows -id $pid]
    set ncol [get_number_of_columns -id $pid]
    puts "  --- $name ---"
    for {set r 0} {$r < $nrow} {incr r} {
        set row ""
        for {set c 0} {$c < $ncol} {incr c} {
            set cell [get_report_panel_data -id $pid -row $r -col $c]
            append row [format "%-40s " $cell]
        }
        puts "    $row"
    }
}

puts ""
puts "=========================================="
puts "Resource usage on EP4CE10F17C8"
puts "=========================================="
dump_panel "Analysis & Synthesis Resource Usage Summary"
puts ""
dump_panel "Analysis & Synthesis Resource Utilization by Entity"

unload_report
project_close

puts ""
puts "Done. Full reports under:"
puts "  $proj_dir/mac_array.map.rpt"
puts "  $proj_dir/mac_array.map.summary"
