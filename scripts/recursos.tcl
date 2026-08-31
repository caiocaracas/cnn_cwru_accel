# mede area do acelerador antes de gastar o fluxo inteiro no SoC.

set nome  [lindex $argv 0]
set raiz  [file normalize [file dirname [info script]]/..]
set parte xc7z020clg400-1
set gen   $raiz/results/gen/$nome

if {![file exists $gen/acelerador_gen.v]} {
    puts "RECURSOS erro sem acelerador_gen.v em $gen"
    exit 1
}

create_project -in_memory -part $parte
add_files -norecurse [concat [glob $raiz/rtl/*.v] [list $gen/acelerador_gen.v]]
set_property file_type SystemVerilog [get_files -quiet */conv1d_engine.v]
set_property file_type SystemVerilog [get_files -quiet */fc_engine.v]

if {[catch {synth_design -top acelerador_gen -part $parte \
                         -mode out_of_context} err]} {
    puts "RECURSOS erro sintese_falhou"
    puts $err
    exit 1
}

set per 8.0
if {$argc > 1} { set per [lindex $argv 1] }
create_clock -period $per -name clk [get_ports clk]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "RECURSOS wns_ns $wns"
puts "RECURSOS fmax_mhz [format %.1f [expr {1000.0/($per-$wns)}]]"

set b36 [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]
set b18 [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]
puts "RECURSOS lut [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]"
puts "RECURSOS ff [llength [get_cells -hier -filter {REF_NAME =~ FD*}]]"
puts "RECURSOS dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]"
puts "RECURSOS bram [expr {$b36 + int(ceil($b18/2.0))}]"
exit 0
