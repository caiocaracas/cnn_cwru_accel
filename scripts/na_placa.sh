#!/usr/bin/env bash
# sobe o sistema na placa por jtag e roda o conjunto de teste.

set -uo pipefail
cd "$(dirname "$0")/.."
SAIDA="${1:-/tmp/placa.txt}"
VIVADO="${2:-/home/caiocv/2025.2/Vivado}"
PETA="${3:-/home/caiocv/petalinux/2025.2}"
PRAZO="${5:-0}"
TTY=/dev/ttyUSB1
CON=$(mktemp /tmp/console.XXXXXX)

cat > /tmp/leitor_placa.sh <<'EOF'
exec cat /dev/ttyUSB1
EOF
chmod +x /tmp/leitor_placa.sh
[ -f /tmp/leitor_placa.pid ] && kill "$(cat /tmp/leitor_placa.pid)" 2>/dev/null
fuser -k /dev/ttyUSB1 2>/dev/null
sleep 1
sleep 1
stty -F $TTY 115200 raw -echo 2>/dev/null
nohup /tmp/leitor_placa.sh > "$CON" 2>&1 &
echo $! > /tmp/leitor_placa.pid

> /tmp/reset_placa.tcl cat <<'EOF'
connect
after 500
configparams force-mem-access 1
targets -set -filter {name =~ "*A9*#0"}
catch { stop }
catch { rst -system }

for {set i 0} {$i < 30} {incr i} {
    after 1000
    if {![catch { targets -filter {name =~ "xc7z*"} } r] && $r ne ""} {
        puts "FPGA na cadeia apos [expr {$i + 1}]s"
        exit 0
    }
}
puts "FPGA nao voltou a cadeia"
exit 1
EOF
source "$VIVADO/settings64.sh" >/dev/null 2>&1
timeout 240 xsdb /tmp/reset_placa.tcl 2>&1 | grep -E "FPGA" || true

echo "programando a placa..."
CFG="${4:-}"
if [ -n "$CFG" ]; then
    BIT="results/vivado/soc_$CFG/$CFG.bit"
    if [ ! -f "$BIT" ]; then
        echo "ERRO: sem bitstream de $CFG; rode a sintese para esta topologia" >&2
        exit 1
    fi
else
    BIT=$(ls -t results/vivado/soc/soc.runs/impl_1/*.bit 2>/dev/null | head -1)
fi
if [ -z "$BIT" ]; then
    echo "ERRO: nenhum bitstream encontrado" >&2
    exit 1
fi
BIT=$(readlink -f "$BIT")
echo "bitstream: $BIT ($(date -r "$BIT" '+%d/%m %H:%M'))"

BOOT_PID=""
for TENT in 1 2 3; do
    : > /tmp/boot_placa.log      # senao a tentativa nova le' o erro da anterior
    ( cd petalinux/cnn_soc && set +u && source "$PETA/settings.sh" >/dev/null 2>&1
      exec timeout 1800 "$PETA/scripts/petalinux-boot" --jtag --kernel --fpga \
        --bitstream "$BIT" ) >/tmp/boot_placa.log 2>&1 &
    BOOT_PID=$!
    for _ in $(seq 1 120); do
        sleep 5
        grep -qE "root@|# $" "$CON" && break
        grep -qE "No supported FPGA device found|non-zero exit status" \
            /tmp/boot_placa.log && break
    done
    grep -qE "root@|# $" "$CON" && break
    echo "  tentativa $TENT falhou: $(grep -m1 -E 'No supported|ERROR' /tmp/boot_placa.log)"
    kill "$BOOT_PID" 2>/dev/null; BOOT_PID=""
    [ "$TENT" = 3 ] && { echo "ERRO: nao consegui programar a placa" >&2; exit 1; }
    sleep 10
done

echo "esperando o sistema subir..."
for _ in $(seq 1 90); do
    sleep 5
    grep -qE "root@|# $" "$CON" && break
done
if ! grep -qE "root@|# $" "$CON"; then
    echo "ERRO: o sistema nao chegou ao console" >&2
    tail -5 "$CON" >&2; exit 1
fi

printf '\r' > $TTY; sleep 2

printf 'echo MARCA_CLK; cat /sys/kernel/debug/clk/fclk0/clk_rate 2>/dev/null; ls /sys/class/fclk 2>/dev/null; cat /sys/class/fclk/fclk0/set_rate 2>/dev/null; echo FIM_CLK\r' > $TTY
sleep 4
sed -n '/MARCA_CLK/,/FIM_CLK/p' "$CON" > "${SAIDA%.txt}.clk.txt"

printf 'for i in /proc/irq/*/smp_affinity; do echo 1 > $i 2>/dev/null; done; echo IRQ_NO_NUCLEO_0\r' > $TTY
sleep 3

printf 'acelerador "" %s > /tmp/r.txt 2>&1; echo TERMINOU=$?\r' "$PRAZO" > $TTY
for _ in $(seq 1 240); do
    sleep 5
    grep -qE "^TERMINOU=" "$CON" && break
done
printf 'cat /tmp/r.txt\r' > $TTY
for _ in $(seq 1 30); do
    sleep 2
    grep -q "ganho de sistema" "$CON" && break
done
sleep 2

[ -n "${BOOT_PID:-}" ] && kill "$BOOT_PID" 2>/dev/null

sed -n '/acelerador CNN 1D/,$p' "$CON" | sed '/^cat \/tmp/d' > "$SAIDA"
grep -q "RESULTADO" "$SAIDA" || { echo "ERRO: saida incompleta" >&2; cat "$SAIDA" >&2; exit 1; }
cat "$SAIDA"
