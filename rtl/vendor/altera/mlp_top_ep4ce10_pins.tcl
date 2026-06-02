# =============================================================================
# 野火征途 Pro (EP4CE10F17C8) pin assignments for mlp_top_demo.
# Pins from the EmbedFire 征途 Pro docs (flow_led example / pin map):
#   doc.embedfire.com/fpga/altera/ep4ce10_pro
#   clk=E1 (50MHz), rst_n=M15 (button), led[0..3]=N3/P3/M6/L7 (4 blue LEDs).
# Verify against your board's "征途引脚绑定映射表.xlsx" before programming.
# =============================================================================

# --- 50 MHz clock ---
set_location_assignment PIN_E1  -to clk
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk

# --- Reset button (active low) ---
set_location_assignment PIN_M15 -to rst_n
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to rst_n

# --- 4 user LEDs -> predicted class in binary (active low on this board) ---
set_location_assignment PIN_N3  -to led_class[0]
set_location_assignment PIN_P3  -to led_class[1]
set_location_assignment PIN_M6  -to led_class[2]
set_location_assignment PIN_L7  -to led_class[3]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_class[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_class[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_class[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to led_class[3]

# --- UART TX → CH340 (115200 8N1) ---
set_location_assignment PIN_B3  -to uart_tx_pin
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to uart_tx_pin
