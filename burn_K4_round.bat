@echo off
"E:\Quart\quartus\bin64\jtagserver.exe" --start
timeout /t 3 /nobreak > /dev/null
"E:\Quart\quartus\bin64\quartus_pgm.exe" -m jtag -o "p;E:\uart_builds\K4_round\demo.sof"
