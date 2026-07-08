# Reservoir Computing Energy Study on SANA-FE

An energy study that compares three reservoir network topologies (random, small-world, and uncoupled) running on a simulated Intel Loihi chip, built on top of the SANA-FE neuromorphic hardware simulator.

## This repository

This repo is a fork of [SANA-FE](https://github.com/SLAM-Lab/SANA-FE), the neuromorphic hardware simulator from SLAM-Lab. All of the upstream source, tests, docs, and tooling are kept intact, and the original credit and license are preserved (see [LICENSE](LICENSE) and the Credits section below).

My own work lives in one place: [`experiments/reservoir/`](experiments/reservoir/). The rest of the tree is upstream SANA-FE and is used as the simulation engine.

## My contribution

I built a small reservoir computer in Python that feeds a sine wave through a spiking reservoir, reads the reservoir state with two simple learners (Ridge regression and an MLP), and then runs the same network through SANA-FE to get energy numbers for a virtual neuromorphic chip. CodeCarbon measures the CPU energy used across the whole run, so each script ends with a chip-versus-CPU energy comparison.

There are three experiments, one per reservoir topology:

- **Random** (`reservoir_random.py`): neurons wired at random with connection density 0.15. Spectral radius scaled to 0.9.
- **Small-world** (`reservoir_smallworld.py`): a Watts-Strogatz network. Neurons sit on a ring, each linked to 4 neighbours per side, with 10% of those edges rewired to distant neurons.
- **Uncoupled** (`reservoir_uncoupled.py`): no recurrent connections at all. Each neuron integrates the input on its own. This is the baseline.

Each script measures two things:

1. How well the readouts reconstruct the input signal (accuracy and mean absolute error), using both membrane-voltage and firing-rate representations of the reservoir state.
2. Energy: SANA-FE's simulated chip energy per sample window (in picojoules) against the CPU energy for the full run (from CodeCarbon).

Every run writes about ten plots and prints the accuracy and energy readings to the terminal. See [`experiments/reservoir/README.md`](experiments/reservoir/README.md) for the per-experiment details.

## Install

You need Python 3 and a few packages:

```
pip install numpy scikit-learn matplotlib pyyaml codecarbon
```

On macOS use `pip3` instead of `pip`. You also need CMake to build the simulator. On macOS you can get it with Homebrew:

```
brew install cmake
```

## Build SANA-FE

The experiments call the compiled SANA-FE binary, so build it once from the repo root:

```
mkdir -p build
cd build
cmake ..
make -j4
```

This produces the simulator at `build/sim`. The experiment scripts look for it there by default. If your binary lives somewhere else, point the scripts at it with the `SANA_FE_BIN` environment variable.

## Run the experiments

Run each script from inside its own folder so the output plots land next to it:

```
cd experiments/reservoir
python3 reservoir_random.py
python3 reservoir_smallworld.py
python3 reservoir_uncoupled.py
```

On macOS use `python3`. On other systems `python` may work.

What to expect from each run:

- Ten PNG plots in the current folder: four signal-versus-prediction plots (Ridge and MLP, for both membrane voltage and firing rate), four matching accuracy-error plots, one chip-energy-per-window plot, and one chip-versus-CPU energy comparison.
- A printed results block with the accuracy and mean absolute error for each readout, the total CPU energy for the run (in kWh, from CodeCarbon), and the SANA-FE chip energy (total and mean per window, in picojoules).

The scripts read the architecture from `experiments/reservoir/simple_reservoir.yaml`. You can point them at a different arch file with the `SANA_FE_ARCH` environment variable.

## Repository layout

```
.
├── experiments/
│   └── reservoir/                  my contribution
│       ├── reservoir_random.py     random topology experiment
│       ├── reservoir_smallworld.py small-world topology experiment
│       ├── reservoir_uncoupled.py  uncoupled baseline experiment
│       ├── simple_reservoir.yaml   SANA-FE arch used by all three
│       ├── results/
│       │   └── powermetrics_log.txt  macOS powermetrics energy capture
│       └── README.md
├── src/            upstream SANA-FE C++ source
├── sanafe/         upstream Python library
├── arch/           upstream example architectures
├── snn/            upstream example networks
├── plugins/        upstream plugins
├── tests/          upstream tests
├── docs/           upstream docs
├── tutorial/       upstream tutorial
├── scripts/        upstream scripts
├── CMakeLists.txt  upstream build config
├── LICENSE         upstream GPL v3 license
└── README.md       this file
```

## Credits and license

The simulator is SANA-FE by SLAM-Lab: https://github.com/SLAM-Lab/SANA-FE. All credit for the simulator goes to the original authors. Their citation details are in [`references.bib`](references.bib).

This repository keeps the upstream license unchanged. SANA-FE is released under the GNU General Public License v3, and the full text is in [LICENSE](LICENSE). The reservoir experiments in `experiments/reservoir/` are my own additions and are covered by the same license.
