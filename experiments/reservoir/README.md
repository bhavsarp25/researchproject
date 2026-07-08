# Reservoir topology energy experiments

Three reservoir computing experiments that run on SANA-FE. Each one feeds a sine wave through a spiking reservoir, reads the state with a Ridge model and an MLP, and compares the simulated chip energy against the CPU energy. The only thing that changes between the three is how the reservoir neurons are wired.

## Files

- `reservoir_random.py` — random topology. Neurons connected at density 0.15, spectral radius scaled to 0.9.
- `reservoir_smallworld.py` — Watts-Strogatz small-world topology. Ring of neurons, 4 neighbours per side, 10% of edges rewired.
- `reservoir_uncoupled.py` — uncoupled baseline. No neuron-to-neuron connections, so each neuron integrates the input on its own.
- `simple_reservoir.yaml` — the SANA-FE architecture all three scripts use. It defines a single-tile, eight-core setup with leaky-integrate-and-fire somas and per-hop energy and latency numbers.
- `results/powermetrics_log.txt` — a macOS `powermetrics` energy capture kept as an experiment artifact.

## Before you run

Build SANA-FE first (see the top-level README). The scripts expect the binary at the repo root under `build/sim`, and they read `simple_reservoir.yaml` from this folder. Both locations can be overridden:

- `SANA_FE_BIN` — path to the compiled `sim` binary.
- `SANA_FE_ARCH` — path to the architecture YAML.

Install the Python deps:

```
pip install numpy scikit-learn matplotlib pyyaml codecarbon
```

## How to run each

Run from this folder so the plots are written here:

```
cd experiments/reservoir

python3 reservoir_random.py
python3 reservoir_smallworld.py
python3 reservoir_uncoupled.py
```

## What each run produces

Ten PNG plots in the current folder:

- `plot1_volt_ridge.png`, `plot2_volt_mlp.png` — signal versus prediction, membrane voltage readout.
- `plot3_rate_ridge.png`, `plot4_rate_mlp.png` — signal versus prediction, firing rate readout.
- `plot5_volt_ridge_accuracy.png`, `plot6_volt_mlp_accuracy.png` — error over time, membrane voltage.
- `plot7_rate_ridge_accuracy.png`, `plot8_rate_mlp_accuracy.png` — error over time, firing rate.
- `plot9_chip_energy.png` — SANA-FE chip energy per sample window during inference.
- `plot10_energy_comparison.png` — chip energy versus CPU energy for the full run.

The terminal also prints a results block: accuracy and mean absolute error for each of the four readouts, total CPU energy for the run (from CodeCarbon), and the SANA-FE chip energy total and mean per window in picojoules.

## A note on scratch files

While running, the scripts write a temporary network file and SANA-FE output under `/tmp` (`/tmp/simple_snn.yaml` and `/tmp/sana_out`). These are scratch files and are safe to delete.
