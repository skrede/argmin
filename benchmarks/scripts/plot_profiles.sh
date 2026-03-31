#!/usr/bin/env bash
# Dolan-More performance profile generator.
#
# Computes and plots performance profiles using f-evals as the primary
# metric.  Failed solvers receive a large performance ratio rather than
# being excluded.
#
# Reference: Dolan & More, "Benchmarking optimization software with
#            performance profiles", Math. Program. 91(2), 2002.
#
# Usage: ./plot_profiles.sh <results.csv> [output_dir]

set -euo pipefail

CSV="${1:?Usage: $0 <results.csv> [output_dir]}"
OUTDIR="${2:-.}"

command -v gnuplot >/dev/null || { echo "gnuplot not found"; exit 1; }
[ -f "$CSV" ] || { echo "File not found: $CSV"; exit 1; }
mkdir -p "$OUTDIR"

# Preprocess CSV into performance ratios per solver, then plot the
# empirical CDF (the performance profile).
#
# Awk reads the CSV, groups by problem to find min f_evals, computes
# ratios, and assigns ratio = 9999 for non-converged runs.

generate_profile() {
    local input_csv="$1"
    local output_pdf="$2"
    local title="$3"

    # Step 1: compute performance ratios with awk.
    local ratios_file
    ratios_file=$(mktemp)

    awk -F, '
    NR == 1 { next }
    {
        solver = $1; problem = $3; fevals = $6; status = $12
        key = solver
        if(!(problem in min_fe) || (fevals + 0) < min_fe[problem])
            min_fe[problem] = fevals + 0
        data[NR] = solver "," problem "," fevals "," status
        solvers[solver] = 1
        n++
    }
    END {
        for(i = 2; i <= NR; i++) {
            if(!(i in data)) continue
            split(data[i], f, ",")
            solver = f[1]; problem = f[2]; fevals = f[3] + 0; status = f[4]
            gsub(/[[:space:]]/, "", status)
            if(status != "converged" && status != "ftol_reached" &&
               status != "xtol_reached")
                ratio = 9999
            else if(min_fe[problem] > 0)
                ratio = fevals / min_fe[problem]
            else
                ratio = 1
            print solver, ratio
        }
    }' "$input_csv" | sort -k1,1 -k2,2n > "$ratios_file"

    # Step 2: identify unique solvers.
    local solvers_file
    solvers_file=$(mktemp)
    awk '{print $1}' "$ratios_file" | sort -u > "$solvers_file"

    # Step 3: create per-solver sorted ratio files for gnuplot.
    local tmpdir
    tmpdir=$(mktemp -d)
    local plot_cmd=""
    local first=1

    while IFS= read -r solver; do
        local sfile="$tmpdir/${solver}.dat"
        awk -v s="$solver" '$1 == s {print $2}' "$ratios_file" | sort -n > "$sfile"

        # Build empirical CDF data: for each ratio value tau, compute
        # fraction of problems where ratio <= tau.
        local cdf_file="$tmpdir/${solver}_cdf.dat"
        awk '
        { vals[NR] = $1; n = NR }
        END {
            for(i = 1; i <= n; i++)
                print vals[i], i / n
        }' "$sfile" > "$cdf_file"

        if [ "$first" -eq 1 ]; then
            plot_cmd="'$cdf_file' using 1:2 with steps lw 2 title '$solver'"
            first=0
        else
            plot_cmd="$plot_cmd, '$cdf_file' using 1:2 with steps lw 2 title '$solver'"
        fi
    done < "$solvers_file"

    # Step 4: plot with gnuplot.
    gnuplot <<GNUPLOT_EOF
set terminal pdfcairo enhanced font "Helvetica,11" size 6,4
set output "$output_pdf"
set xlabel "{/Symbol t}"
set ylabel "{/Symbol r}_s({/Symbol t})"
set title "$title"
set key bottom right
set grid
set xrange [1:*]
set yrange [0:1.05]
set logscale x 2
plot $plot_cmd
GNUPLOT_EOF

    echo "Generated: $output_pdf"
    rm -rf "$tmpdir" "$ratios_file" "$solvers_file"
}

# Overall performance profile.
generate_profile "$CSV" "$OUTDIR/performance_profile.pdf" \
    "Performance Profile (f-evals)"

# Per-class breakdown profiles (RPT-03).
for CLASS in unconstrained bound_constrained inequality equality global; do
    class_csv=$(mktemp)
    head -1 "$CSV" > "$class_csv"
    awk -F, -v c="$CLASS" 'NR > 1 && index($4, c)' "$CSV" >> "$class_csv"

    row_count=$(wc -l < "$class_csv")
    if [ "$row_count" -gt 1 ]; then
        generate_profile "$class_csv" \
            "$OUTDIR/profile_${CLASS}.pdf" \
            "Performance Profile: ${CLASS} (f-evals)"
    else
        echo "No data for class: $CLASS, skipping."
    fi
    rm -f "$class_csv"
done
