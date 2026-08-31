# monta o bloco do SoC, sintetiza e gera o bitstream.

set clk_mhz [lindex $argv 0]
set cfgnome [lindex $argv 1]
set so_bd   [lindex $argv 2]
if {$clk_mhz eq ""} { set clk_mhz 100 }
if {$cfgnome eq ""} { set cfgnome L3_F08_K5_Pmax }

set raiz  [file normalize [file dirname [info script]]/..]
set parte xc7z020clg400-1
set proj  $raiz/results/vivado/soc

set_param board.repoPaths [list $raiz/board_files]

file delete -force $proj
create_project -force soc $proj -part $parte
set bp [lindex [get_board_parts -quiet *arty-z7-20*] 0]
if {$bp eq ""} { puts "ERRO: board part arty-z7-20 nao encontrado"; exit 1 }
puts "usando board part: $bp"
set_property board_part $bp [current_project]

set gerado $raiz/results/gen/$cfgnome/acelerador_gen.v
if {![file exists $gerado]} {
    puts "ERRO: $gerado nao existe - rode o gerador antes"
    exit 1
}
set fontes {}
foreach f [glob $raiz/rtl/*.v] {
    if {[file tail $f] ne "acelerador.v"} { lappend fontes $f }
}
lappend fontes $gerado
add_files -norecurse $fontes
puts "usando microarquitetura gerada: $cfgnome"

set_property file_type SystemVerilog [get_files -quiet */conv1d_engine.v]
set_property file_type SystemVerilog [get_files -quiet */fc_engine.v]
update_compile_order -fileset sources_1

create_bd_design sistema

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config [list make_external "FIXED_IO, DDR" apply_board_preset "1" \
                  Master "Disable" Slave "Disable"] [get_bd_cells ps7]

set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $clk_mhz] [get_bd_cells ps7]

create_bd_cell -type module -reference axi_acelerador acel
set fluxo 0
if {[file exists $raiz/results/gen/$cfgnome/plano.json]} {
    set fh [open $raiz/results/gen/$cfgnome/plano.json r]
    set txt [read $fh]; close $fh
    if {[regexp {\"fluxo\"\s*:\s*true} $txt]} { set fluxo 1 }
}
set_property -dict [list CONFIG.FLUXO $fluxo] [get_bd_cells acel]
puts "ENVELOPE fluxo=$fluxo"

set ifs [get_bd_intf_pins acel/*]
puts "INTF inferidas: $ifs"
set alvo ""
set alvo_s ""
foreach p $ifs {
    if {[string match "*axis*" [string tolower $p]]} {
        set alvo_s $p
    } elseif {[string match "*axi*" [string tolower $p]]} {
        set alvo $p
    }
}
if {$alvo eq ""} {
    puts "ERRO: Vivado nao inferiu a interface AXI no module reference"
    exit 1
}
if {$alvo_s eq ""} {
    puts "ERRO: Vivado nao inferiu a interface AXI-Stream no module reference"
    exit 1
}
puts "usando interface: $alvo (controle) e $alvo_s (janela)"

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Master "/ps7/M_AXI_GP0" Clk "Auto"] [get_bd_intf_pins $alvo]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma dma
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_mm2s_dre {0} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_mm2s_burst_size {16}] [get_bd_cells dma]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Master "/ps7/M_AXI_GP0" Clk "Auto"] \
    [get_bd_intf_pins dma/S_AXI_LITE]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Master "/dma/M_AXI_MM2S" Slave "/ps7/S_AXI_HP0" \
                  ddr_seg "Auto" intc_ip "Auto" Clk_xbar "Auto" \
                  Clk_master "Auto" Clk_slave "Auto"] \
    [get_bd_intf_pins ps7/S_AXI_HP0]

connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins $alvo_s]

create_bd_port -dir O -from 3 -to 0 leds
connect_bd_net [get_bd_pins acel/leds] [get_bd_ports leds]

assign_bd_address
validate_bd_design
save_bd_design
regenerate_bd_layout

foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data]] {
    puts "RESUMO endereco [get_property NAME $seg] [format 0x%08X \
          [get_property OFFSET $seg]]"
}

if {$so_bd ne ""} {
    puts "RESUMO bd       validado, parando antes da implementacao"
    exit 0
}

make_wrapper -files [get_files sistema.bd] -top
add_files -norecurse $proj/soc.gen/sources_1/bd/sistema/hdl/sistema_wrapper.v
add_files -fileset constrs_1 -norecurse $raiz/scripts/leds.xdc

set margem_ns [format %.3f [expr {0.05 * 1000.0 / $clk_mhz}]]
set fx [open $proj/margem.xdc w]
puts $fx "# gerado por scripts/bitstream.tcl: 5% de $clk_mhz MHz"
puts $fx "set_clock_uncertainty -setup $margem_ns \[get_clocks clk_fpga_0\]"
close $fx
add_files -fileset constrs_1 -norecurse $proj/margem.xdc
puts "MARGEM incerteza de $margem_ns ns pedida em clk_fpga_0"
set_property top sistema_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERRO: implementacao falhou"
    puts [get_property STATUS [get_runs impl_1]]
    exit 1
}

set faltando 0
foreach reg [glob -nocomplain $proj/soc.runs/*/runme.log] {
    set fh [open $reg r]
    foreach linha [split [read $fh] "\n"] {
        if {[string match "*Synth 8-4445*" $linha]} { incr faltando }
    }
    close $fh
}
if {$faltando > 0} {
    puts "RESUMO memorias  $faltando arquivos de inicializacao nao lidos"
    puts "NAO ENTREGA: banco de memoria vazio"
    exit 3
}

open_run impl_1
set saida $raiz/results/vivado/soc_$cfgnome
file mkdir $saida
report_utilization     -file $saida/util.rpt

write_hw_platform -fixed -include_bit -force $saida/sistema.xsa
report_timing_summary  -file $saida/timing.rpt
report_power           -file $saida/potencia.rpt

report_methodology -file $saida/metodologia.rpt -quiet
set metod 0
if {[file exists $saida/metodologia.rpt]} {
    set fh [open $saida/metodologia.rpt r]
    foreach ln [split [read $fh] "\n"] {
        if {[regexp {^\| +[0-9]+ +\| +([A-Z]+-[0-9]+)} $ln -> regra]} {
            incr metod
            puts "RESUMO metodologia_$metod $regra"
        }
    }
    close $fh
}
puts "RESUMO metodologia $metod violacoes"

set wns [get_property SLACK [get_timing_paths -delay_type max]]
set whs [get_property SLACK [get_timing_paths -delay_type min]]

set pc [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]
if {$pc ne ""} {
    puts "RESUMO caminho_de   [get_property STARTPOINT_PIN $pc]"
    puts "RESUMO caminho_para [get_property ENDPOINT_PIN $pc]"
    puts "RESUMO caminho_niveis [get_property LOGIC_LEVELS $pc]"
}

set relogio [get_clocks -of_objects [get_pins -hier -filter {NAME =~ *FCLK_CLK0}]]
set per [get_property PERIOD $relogio]

report_timing -max_paths 1 -nworst 1 -setup -file $saida/caminho.rpt
set unc 0.0
if {[file exists $saida/caminho.rpt]} {
    set fh [open $saida/caminho.rpt r]
    foreach ln [split [read $fh] "\n"] {
        if {[regexp {clock uncertainty\s+(-?[0-9.]+)} $ln -> v]} {
            set unc [expr {abs($v)}]
        }
    }
    close $fh
}
set unc_min [expr {0.05 * $per * 0.99}]
puts "RESUMO incerteza $unc ns  (pedida [format %.3f [expr {0.05*$per}]])"
if {$unc < $unc_min} {
    puts "RESUMO fecha     nao"
    puts "NAO FECHA: a margem de tempo nao entrou na analise (incerteza $unc ns\
 no caminho critico); confira o XDC gerado em $proj/margem.xdc"
    exit 3
}
set clk_real [format %.3f [expr {1000.0/$per}]]
puts "RESUMO clock    $clk_real MHz"
puts "RESUMO pedido   $clk_mhz MHz"
puts "RESUMO wns      $wns ns"
puts "RESUMO fmax     [format %.1f [expr {1000.0/($per - $wns)}]] MHz"
puts "RESUMO fmax_seguro [format %.1f [expr {1000.0/($per - $wns)}]] MHz"
puts "RESUMO dsp      [llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]"
puts "RESUMO lut      [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]"
puts "RESUMO ff       [llength [get_cells -hier -filter {REF_NAME =~ FD*}]]"
puts "RESUMO bram36   [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]"
puts "RESUMO bram18   [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]"

puts "RESUMO whs      $whs ns"
puts "RESUMO margem_embutida [format %.3f [expr {0.05*$per}]] ns"

if {$wns < 0} {
    puts "RESUMO fecha     nao"
    puts "NAO FECHA: folga $wns ns em $clk_real MHz"
    exit 2
}
if {$whs < 0} {
    puts "RESUMO fecha     nao"
    puts "NAO FECHA: folga de hold $whs ns em $clk_real MHz"
    exit 2
}
puts "RESUMO fecha     sim"

set crit_nosso 0
set crit_placa 0
foreach arq [list $proj/soc.runs/synth_1/runme.log $proj/soc.runs/impl_1/runme.log] {
    if {![file exists $arq]} { continue }
    set fh [open $arq r]
    foreach ln [split [read $fh] "\n"] {
        if {[string match "CRITICAL WARNING*" $ln]} {
            if {[string match "*PSU-*" $ln] || [string match "*DDR*" $ln]} {
                incr crit_placa
            } else {
                incr crit_nosso
                puts "RESUMO critico_nosso [string range $ln 0 160]"
            }
        }
    }
    close $fh
}
puts "RESUMO criticos_do_desenho $crit_nosso"
puts "RESUMO criticos_do_preset  $crit_placa"
if {$crit_nosso > 0} {
    puts "RESUMO fecha     nao"
    puts "NAO FECHA: $crit_nosso avisos criticos vindos do desenho gerado"
    exit 4
}

set bit [lindex [glob -nocomplain $proj/soc.runs/impl_1/*.bit] 0]

set guardado $saida/$cfgnome.bit
file copy -force $bit $guardado
puts "RESUMO bitstream $guardado"
exit 0
