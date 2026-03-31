#!/usr/bin/env bash
# Convergence trace plotter.
#
# Reads per-solver/problem trace CSV files produced by the benchmark
# driver and generates convergence plots (iteration vs objective value).
#
# Usage: ./plot_convergence.sh <trace_dir> [output_dir]

set -euo pipefail

TRACE_DIR="${1:?Usage: $0 <trace_dir> [output_dir]}"
OUTDIR="${2:-.}"

command -v gnuplot >/dev/null || { echo "gnuplot not found"; exit 1; }
[ -d "$TRACE_DIR" ] || { echo "Directory not found: $TRACE_DIR"; exit 1; }
mkdir -p "$OUTDIR"

count=0

for csv in "$TRACE_DIR"/*.csv; do
    [ -f "$csv" ] || continue

    basename_no_ext=$(basename "$csv" .csv)
    output_pdf="$OUTDIR/convergence_${basename_no_ext}.pdf"

    # Determine if log scale is appropriate: check if objective values
    # span more than two orders of magnitude and are all positive.
    use_log=$(awk -F, '
    NR == 1 { next }
    {
        v = $2 + 0
        if(v <= 0) { print "no"; exit }
        if(NR == 2) { mn = v; mx = v }
        if(v < mn) mn = v
        if(v > mx) mx = v
    }
    END {
        if(mn > 0 && mx / mn > 100) print "yes"
        else print "no"
    }' "$csv")

    logscale_cmd=""
    if [ "$use_log" = "yes" ]; then
        logscale_cmd="set logscale y"
    fi

    gnuplot <<GNUPLOT_EOF
set terminal pdfcairo enhanced font "Helvetica,11" size 6,4
set output "$output_pdf"
set xlabel "Iteration"
set ylabel "Objective Value"
set title "Convergence: ${basename_no_ext}"
set grid
set datafile separator ","
$logscale_cmd
plot "$csv" using 1:2 every ::1 with lines lw 2 notitle
GNUPLOT_EOF

    count=$((count + 1))
done

echo "Generated $count convergence plots in $OUTDIR"
