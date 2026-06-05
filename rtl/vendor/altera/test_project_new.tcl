load_package flow
set rtl_root [pwd]
puts "PWD: $rtl_root"
set pdir "$rtl_root/vendor/altera/build/uart_test_minimal"
file mkdir $pdir
cd $pdir
puts "Creating project in: [pwd]"
project_new testproj -overwrite
puts "Project created. Files: [glob -nocomplain *]"
project_close
puts "DONE"
