"""
Simple Reservoir Computer — Uncoupled topology + SANA-FE
=========================================================
A sine wave goes in. The reservoir (uncoupled — no recurrent
connections) processes it in Python. SANA-FE simulates the same
network on virtual neuromorphic hardware and gives us real chip
energy numbers. Ridge and MLP read the state using both membrane
voltage and firing rate representations. CodeCarbon tracks CPU
energy across the entire run.

Reservoir topology: uncoupled. Each neuron independently integrates
the input signal. W is all zeros — no neuron-to-neuron connections.
This is the baseline — any accuracy comes from input encoding alone.

Outputs:
  plot1_volt_ridge.png          signal vs Ridge (membrane voltage)
  plot2_volt_mlp.png            signal vs MLP   (membrane voltage)
  plot3_rate_ridge.png          signal vs Ridge (firing rate)
  plot4_rate_mlp.png            signal vs MLP   (firing rate)
  plot5_volt_ridge_accuracy.png Ridge error     (membrane voltage)
  plot6_volt_mlp_accuracy.png   MLP error       (membrane voltage)
  plot7_rate_ridge_accuracy.png Ridge error     (firing rate)
  plot8_rate_mlp_accuracy.png   MLP error       (firing rate)
  plot9_chip_energy.png         SANA-FE energy per window
  plot10_energy_comparison.png  chip vs CPU energy

Run:  python3 reservoir_uncoupled.py
Need: pip install numpy scikit-learn matplotlib pyyaml codecarbon
      SANA-FE built at ~/SANA-FE/build/sim
"""

import os
import re
import subprocess

import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.linear_model   import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from codecarbon             import EmissionsTracker


# ── Settings ──────────────────────────────────────────────────
N_NEURONS    = 64      # neurons in the reservoir
N_INPUT      = 8       # input encoder neurons
LEAK         = 0.9     # membrane voltage leak per step
THRESHOLD    = 1.0     # voltage needed to fire
RATE_WIN     = 10      # timesteps to average spikes over for firing rate
WARMUP_SECS  = 6.0     # warmup duration
TRAIN_SECS   = 6.0     # training duration
INFER_SECS   = 10.0    # inference duration
SAMPLE_RATE  = 40      # samples per second
WINDOW_T     = 30      # SANA-FE timesteps per sample window
SEED         = 42

# SANA-FE paths
SANA_BIN  = os.path.expanduser("~/SANA-FE/build/sim")
ARCH_FILE = os.path.expanduser("~/SANA-FE/arch/simple_reservoir.yaml")
SNN_FILE  = "/tmp/simple_snn.yaml"
OUT_DIR   = "/tmp/sana_out"


# ── Step 1: Signal ─────────────────────────────────────────────
def live_signal(t):
    return np.sin(2 * np.pi * 0.3 * t)


# ── Step 2: Auto-scaling encoder ───────────────────────────────
class AutoEncoder:
    def __init__(self, n=8):
        self.n       = n
        self.centers = np.linspace(-1.0, 1.0, n)
        self.sigma   = 2.0 / max(n - 1, 1)
        self._seen   = []

    def observe(self, value):
        self._seen.append(float(value))

    def calibrate(self):
        lo = min(self._seen)
        hi = max(self._seen)
        margin = max((hi - lo) * 0.1, 0.05)
        lo -= margin
        hi += margin
        self.centers = np.linspace(lo, hi, self.n)
        self.sigma   = (hi - lo) / max(self.n - 1, 1)
        print(f"  Encoder calibrated: [{lo:.3f}, {hi:.3f}]")

    def encode(self, value):
        response = np.exp(-0.5 * ((value - self.centers) / self.sigma) ** 2)
        return 0.15 + 0.85 * response

encoder = AutoEncoder(n=N_INPUT)


# ── Step 3: Reservoir — Uncoupled topology ─────────────────────
# W is all zeros — no neuron-to-neuron connections.
# Each neuron independently integrates the input via W_in only.
rng = np.random.RandomState(SEED)
W   = np.zeros((N_NEURONS, N_NEURONS))

W_in    = rng.randn(N_NEURONS, N_INPUT) * 0.1
voltage = np.zeros(N_NEURONS)

# Spike history buffer for firing rate (circular buffer, RATE_WIN steps)
spike_history = np.zeros((RATE_WIN, N_NEURONS))
history_idx   = 0

def reservoir_step(encoded):
    """
    Run one timestep. Returns:
      volt_state : membrane voltages (64,)
      rate_state : mean firing rate over last RATE_WIN steps (64,)
    """
    global voltage, spike_history, history_idx

    drive   = W_in @ encoded + W @ (voltage > 0).astype(float)
    voltage = LEAK * voltage + drive
    spikes  = (voltage >= THRESHOLD).astype(float)
    voltage[spikes > 0] = 0.0

    spike_history[history_idx] = spikes
    history_idx = (history_idx + 1) % RATE_WIN
    rate_state  = spike_history.mean(axis=0)

    return voltage.copy(), rate_state.copy()


# ── Step 4: SANA-FE interface ──────────────────────────────────

def write_snn_yaml(encoded):
    os.makedirs("/tmp", exist_ok=True)
    NPC  = N_NEURONS // 8
    seed = int(np.sum(encoded) * 1000) % 100000
    rng2 = np.random.RandomState(seed)
    input_spikes = (rng2.rand(N_INPUT, WINDOW_T) < encoded[:, None]).astype(int)

    with open(SNN_FILE, "w") as f:
        f.write("network:\n")
        f.write("  name: simple_reservoir\n")
        f.write("  groups:\n")

        f.write("    - name: input_group\n")
        f.write("      attributes: []\n")
        f.write("      neurons:\n")
        for i in range(N_INPUT):
            spk = [int(x) for x in input_spikes[i]]
            f.write(f"        - {i}: [spikes: {spk}]\n")

        f.write("    - name: res_group\n")
        f.write("      attributes: []\n")
        f.write("      neurons:\n")
        for i in range(N_NEURONS):
            f.write(f"        - {i}: [soma: {{threshold: {THRESHOLD}}}]\n")

        f.write("  edges:\n")
        for pre in range(N_INPUT):
            for post in range(N_NEURONS):
                w = float(W_in[post, pre])
                if abs(w) > 0.001:
                    f.write(f"    - input_group.{pre} -> res_group.{post}"
                            f": [weight: {w:.4f}]\n")
        # W is all zeros for uncoupled — no recurrent edges written

        f.write("mappings:\n")
        for i in range(N_INPUT):
            f.write(f"  - input_group.{i}: [core: 0.0, soma: input_soma]\n")
        for i in range(N_NEURONS):
            core = i // NPC
            f.write(f"  - res_group.{i}: [core: 0.{core}, soma: res_soma]\n")


def run_sana_fe():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd    = [SANA_BIN, "-p", "-s", "-o", OUT_DIR,
              ARCH_FILE, SNN_FILE, str(WINDOW_T)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return 0.0

    summary_path = os.path.join(OUT_DIR, "run_summary.yaml")
    if os.path.exists(summary_path):
        try:
            raw   = open(summary_path).read()
            fixed = re.sub(r'([a-zA-Z0-9_]):(\S)', r'\1: \2', raw)
            data  = yaml.safe_load(fixed) or {}
            total = data.get("energy", {}).get("total", 0.0)
            return float(total) * 1e12
        except Exception:
            pass
    return 0.0


# ── Step 5: Main loop ──────────────────────────────────────────
dt           = 1.0 / SAMPLE_RATE
warmup_steps = int(WARMUP_SECS / dt)
train_steps  = int(TRAIN_SECS  / dt)
infer_steps  = int(INFER_SECS  / dt)
total_steps  = warmup_steps + train_steps + infer_steps

times        = []
signals      = []
sana_energy  = []

# Membrane voltage readout lists
volt_ridge_preds  = []
volt_mlp_preds    = []
train_volt_states = []
train_targets     = []

# Firing rate readout lists
rate_ridge_preds  = []
rate_mlp_preds    = []
train_rate_states = []

volt_ridge_model  = None
volt_mlp_model    = None
rate_ridge_model  = None
rate_mlp_model    = None

volt_ridge_scaler = StandardScaler()
volt_mlp_scaler   = StandardScaler()
rate_ridge_scaler = StandardScaler()
rate_mlp_scaler   = StandardScaler()

# CodeCarbon — track CPU energy for the entire run
cpu_tracker = EmissionsTracker(log_level="error", save_to_file=False,
                               measure_power_secs=999)
cpu_tracker.start()

print(f"Running {total_steps} steps "
      f"({WARMUP_SECS}s warmup / {TRAIN_SECS}s train / {INFER_SECS}s infer)")
print(f"SANA-FE: {SANA_BIN}")
print(f"Arch:    {ARCH_FILE}\n")

for step in range(total_steps):
    t   = step * dt
    val = live_signal(t)

    # Determine phase
    if step < warmup_steps:
        phase = "warmup"
    elif step < warmup_steps + train_steps:
        phase = "train"
    else:
        phase = "infer"

    # Encoder observe + calibrate
    if phase == "warmup":
        encoder.observe(val)
    if step == warmup_steps - 1:
        encoder.calibrate()

    encoded = encoder.encode(val)

    # Run reservoir — get both voltage and rate states
    volt_state, rate_state = reservoir_step(encoded)

    # Collect training data
    if phase == "train":
        train_volt_states.append(volt_state.copy())
        train_rate_states.append(rate_state.copy())
        train_targets.append(val)

    # Fit all four readouts at end of training
    if step == warmup_steps + train_steps - 1:
        y = np.array(train_targets)

        # ── Voltage readouts ──
        Xv = volt_ridge_scaler.fit_transform(np.array(train_volt_states))

        volt_ridge_model = Ridge(alpha=0.001)
        volt_ridge_model.fit(Xv, y)
        print(f"  Voltage Ridge fitted")

        vr_train = volt_ridge_model.predict(Xv)
        Xv_mlp   = np.column_stack([np.array(train_volt_states), vr_train])
        Xv_mlp   = volt_mlp_scaler.fit_transform(Xv_mlp)
        volt_mlp_model = MLPRegressor(hidden_layer_sizes=(32, 16),
                                      activation="tanh", max_iter=500,
                                      random_state=SEED)
        volt_mlp_model.fit(Xv_mlp, y)
        print(f"  Voltage MLP   fitted")

        # ── Rate readouts ──
        Xr = rate_ridge_scaler.fit_transform(np.array(train_rate_states))

        rate_ridge_model = Ridge(alpha=0.001)
        rate_ridge_model.fit(Xr, y)
        print(f"  Rate    Ridge fitted")

        rr_train = rate_ridge_model.predict(Xr)
        Xr_mlp   = np.column_stack([np.array(train_rate_states), rr_train])
        Xr_mlp   = rate_mlp_scaler.fit_transform(Xr_mlp)
        rate_mlp_model = MLPRegressor(hidden_layer_sizes=(32, 16),
                                      activation="tanh", max_iter=500,
                                      random_state=SEED)
        rate_mlp_model.fit(Xr_mlp, y)
        print(f"  Rate    MLP   fitted")

    # Predict from both readout types
    sig_lo = float(encoder.centers[0])
    sig_hi = float(encoder.centers[-1])

    vr_pred, vm_pred = 0.0, 0.0
    rr_pred, rm_pred = 0.0, 0.0

    if volt_ridge_model is not None:
        # Voltage predictions
        xvs     = volt_ridge_scaler.transform(volt_state.reshape(1, -1))
        vr_pred = float(np.clip(volt_ridge_model.predict(xvs)[0], sig_lo, sig_hi))
        xvm     = volt_mlp_scaler.transform(np.append(volt_state, vr_pred).reshape(1, -1))
        vm_pred = float(np.clip(volt_mlp_model.predict(xvm)[0], sig_lo, sig_hi))

        # Rate predictions
        xrs     = rate_ridge_scaler.transform(rate_state.reshape(1, -1))
        rr_pred = float(np.clip(rate_ridge_model.predict(xrs)[0], sig_lo, sig_hi))
        xrm     = rate_mlp_scaler.transform(np.append(rate_state, rr_pred).reshape(1, -1))
        rm_pred = float(np.clip(rate_mlp_model.predict(xrm)[0], sig_lo, sig_hi))

    # SANA-FE energy — inference only
    if phase == "infer":
        write_snn_yaml(encoded)
        chip_pj = run_sana_fe()
    else:
        chip_pj = 0.0

    times.append(t)
    signals.append(val)
    sana_energy.append(chip_pj)
    volt_ridge_preds.append(vr_pred)
    volt_mlp_preds.append(vm_pred)
    rate_ridge_preds.append(rr_pred)
    rate_mlp_preds.append(rm_pred)

    if step % 20 == 0:
        print(f"  [{phase:6s}] t={t:.2f}s | sig={val:+.3f} | "
              f"v_ridge={vr_pred:+.3f} | r_ridge={rr_pred:+.3f} | "
              f"chip={chip_pj:.0f} pJ")

print("\nDone.")

# Stop CPU tracker
cpu_tracker.stop()
cpu_kwh = cpu_tracker._total_energy.kWh
print(f"  CPU total energy (full run): {cpu_kwh:.8f} kWh")


# ── Step 6: Accuracy ───────────────────────────────────────────
infer_start = warmup_steps + train_steps

def stats(preds, targets):
    p, t = np.array(preds), np.array(targets)
    mae  = np.mean(np.abs(p - t))
    r2   = 1 - np.sum((p-t)**2) / max(np.sum((t-np.mean(t))**2), 1e-9)
    return max(0.0, r2), mae

vr_r2, vr_mae = stats(volt_ridge_preds[infer_start:], signals[infer_start:])
vm_r2, vm_mae = stats(volt_mlp_preds[infer_start:],   signals[infer_start:])
rr_r2, rr_mae = stats(rate_ridge_preds[infer_start:], signals[infer_start:])
rm_r2, rm_mae = stats(rate_mlp_preds[infer_start:],   signals[infer_start:])

infer_energies = [e for e in sana_energy[infer_start:] if e > 0]
sana_total_pj  = sum(infer_energies)
sana_mean_pj   = np.mean(infer_energies) if infer_energies else 0.0

print("\n" + "="*55)
print("  RESULTS")
print("="*55)
print(f"  Voltage Ridge — Accuracy: {vr_r2*100:.1f}%  MAE: {vr_mae:.4f}")
print(f"  Voltage MLP   — Accuracy: {vm_r2*100:.1f}%  MAE: {vm_mae:.4f}")
print(f"  Rate    Ridge — Accuracy: {rr_r2*100:.1f}%  MAE: {rr_mae:.4f}")
print(f"  Rate    MLP   — Accuracy: {rm_r2*100:.1f}%  MAE: {rm_mae:.4f}")
print(f"  CPU total (full run): {cpu_kwh:.8f} kWh")
print(f"  SANA-FE — Total: {sana_total_pj:.1f} pJ  Mean/window: {sana_mean_pj:.1f} pJ")
print("="*55)


# ── Step 7: Plots ──────────────────────────────────────────────
warmup_t = warmup_steps * dt
train_t  = (warmup_steps + train_steps) * dt
infer_t  = times[infer_start:]

def vlines(ax):
    ax.axvline(warmup_t, color="grey", ls="--", lw=1.0)
    ax.axvline(train_t,  color="grey", ls="--", lw=1.0)
    ax.text(warmup_t/2,            1.05, "WARMUP",
            ha="center", fontsize=8, color="grey",
            transform=ax.get_xaxis_transform())
    ax.text((warmup_t+train_t)/2,  1.05, "TRAINING",
            ha="center", fontsize=8, color="grey",
            transform=ax.get_xaxis_transform())
    ax.text((train_t+times[-1])/2, 1.05, "INFERENCE",
            ha="center", fontsize=8, color="grey",
            transform=ax.get_xaxis_transform())

# ── 4 signal vs prediction plots ──────────────────────────────

# Plot 1 — Signal vs Voltage Ridge
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, signals,          color="black", lw=2.0, label="Signal")
ax.plot(times, volt_ridge_preds, color="red",   lw=1.5, label="Ridge")
vlines(ax)
ax.legend(fontsize=10); ax.grid(alpha=0.2)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_title("Signal vs Ridge Prediction — Membrane Voltage (Uncoupled)")
fig.tight_layout(); fig.savefig("plot1_volt_ridge.png", dpi=150); plt.close()

# Plot 2 — Signal vs Voltage MLP
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, signals,        color="black", lw=2.0, label="Signal")
ax.plot(times, volt_mlp_preds, color="blue",  lw=1.5, label="MLP")
vlines(ax)
ax.legend(fontsize=10); ax.grid(alpha=0.2)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_title("Signal vs MLP Prediction — Membrane Voltage (Uncoupled)")
fig.tight_layout(); fig.savefig("plot2_volt_mlp.png", dpi=150); plt.close()

# Plot 3 — Signal vs Rate Ridge
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, signals,          color="black",      lw=2.0, label="Signal")
ax.plot(times, rate_ridge_preds, color="darkorange", lw=1.5, label="Ridge")
vlines(ax)
ax.legend(fontsize=10); ax.grid(alpha=0.2)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_title("Signal vs Ridge Prediction — Firing Rate (Uncoupled)")
fig.tight_layout(); fig.savefig("plot3_rate_ridge.png", dpi=150); plt.close()

# Plot 4 — Signal vs Rate MLP
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, signals,        color="black",  lw=2.0, label="Signal")
ax.plot(times, rate_mlp_preds, color="purple", lw=1.5, label="MLP")
vlines(ax)
ax.legend(fontsize=10); ax.grid(alpha=0.2)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_title("Signal vs MLP Prediction — Firing Rate (Uncoupled)")
fig.tight_layout(); fig.savefig("plot4_rate_mlp.png", dpi=150); plt.close()

# ── Accuracy error plots ───────────────────────────────────────
w = min(10, infer_steps)

# Plot 5 — Voltage Ridge accuracy
err_vr = np.abs(np.array(volt_ridge_preds[infer_start:]) - np.array(signals[infer_start:]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_t[w-1:], np.convolve(err_vr, np.ones(w)/w, mode="valid"),
        color="red", lw=1.5, label="Rolling MAE")
ax.axhline(vr_mae, color="black", ls="--", lw=1, label=f"Mean={vr_mae:.3f}")
ax.text(0.97, 0.95, f"Accuracy: {vr_r2*100:.1f}%\nR2: {vr_r2:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.4"))
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Absolute Error")
ax.set_title("Ridge Accuracy — Membrane Voltage (Uncoupled)")
fig.tight_layout(); fig.savefig("plot5_volt_ridge_accuracy.png", dpi=150); plt.close()

# Plot 6 — Voltage MLP accuracy
err_vm = np.abs(np.array(volt_mlp_preds[infer_start:]) - np.array(signals[infer_start:]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_t[w-1:], np.convolve(err_vm, np.ones(w)/w, mode="valid"),
        color="blue", lw=1.5, label="Rolling MAE")
ax.axhline(vm_mae, color="black", ls="--", lw=1, label=f"Mean={vm_mae:.3f}")
ax.text(0.97, 0.95, f"Accuracy: {vm_r2*100:.1f}%\nR2: {vm_r2:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.4"))
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Absolute Error")
ax.set_title("MLP Accuracy — Membrane Voltage (Uncoupled)")
fig.tight_layout(); fig.savefig("plot6_volt_mlp_accuracy.png", dpi=150); plt.close()

# Plot 7 — Rate Ridge accuracy
err_rr = np.abs(np.array(rate_ridge_preds[infer_start:]) - np.array(signals[infer_start:]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_t[w-1:], np.convolve(err_rr, np.ones(w)/w, mode="valid"),
        color="darkorange", lw=1.5, label="Rolling MAE")
ax.axhline(rr_mae, color="black", ls="--", lw=1, label=f"Mean={rr_mae:.3f}")
ax.text(0.97, 0.95, f"Accuracy: {rr_r2*100:.1f}%\nR2: {rr_r2:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.4"))
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Absolute Error")
ax.set_title("Ridge Accuracy — Firing Rate (Uncoupled)")
fig.tight_layout(); fig.savefig("plot7_rate_ridge_accuracy.png", dpi=150); plt.close()

# Plot 8 — Rate MLP accuracy
err_rm = np.abs(np.array(rate_mlp_preds[infer_start:]) - np.array(signals[infer_start:]))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(infer_t[w-1:], np.convolve(err_rm, np.ones(w)/w, mode="valid"),
        color="purple", lw=1.5, label="Rolling MAE")
ax.axhline(rm_mae, color="black", ls="--", lw=1, label=f"Mean={rm_mae:.3f}")
ax.text(0.97, 0.95, f"Accuracy: {rm_r2*100:.1f}%\nR2: {rm_r2:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.4"))
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Absolute Error")
ax.set_title("MLP Accuracy — Firing Rate (Uncoupled)")
fig.tight_layout(); fig.savefig("plot8_rate_mlp_accuracy.png", dpi=150); plt.close()

# ── Energy plots ───────────────────────────────────────────────

# Plot 9 — SANA-FE chip energy per inference step
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(times[infer_start:], sana_energy[infer_start:],
       width=dt*0.9, color="steelblue", alpha=0.7)
if sana_mean_pj > 0:
    ax.axhline(sana_mean_pj, color="black", ls="--", lw=1,
               label=f"Mean = {sana_mean_pj:.0f} pJ")
    ax.legend()
ax.grid(alpha=0.2, axis="y")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (pJ)")
ax.set_title("SANA-FE Chip Energy per Sample Window — Uncoupled (Inference)")
fig.tight_layout(); fig.savefig("plot9_chip_energy.png", dpi=150); plt.close()

# Plot 10 — Energy comparison: chip vs CPU
fig, ax = plt.subplots(figsize=(7, 5))
sana_nj = sana_total_pj / 1000
cpu_nj  = cpu_kwh * 3.6e12 / 1000
vals    = [sana_nj, cpu_nj]
bars    = ax.bar(["SANA-FE\n(chip)", "CPU\n(full run)"],
                 vals, color=["steelblue", "tomato"], alpha=0.8, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(vals) * 0.02,
            f"{v:.3f} nJ", ha="center", fontsize=9)
ax.set_ylabel("Energy (nJ)")
ax.set_title("Energy Comparison: Chip vs CPU — Uncoupled (Full Run)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("plot10_energy_comparison.png", dpi=150); plt.close()

print("\nPlots saved:")
print("  plot1_volt_ridge.png          signal vs Ridge (membrane voltage)")
print("  plot2_volt_mlp.png            signal vs MLP   (membrane voltage)")
print("  plot3_rate_ridge.png          signal vs Ridge (firing rate)")
print("  plot4_rate_mlp.png            signal vs MLP   (firing rate)")
print("  plot5_volt_ridge_accuracy.png Ridge error     (membrane voltage)")
print("  plot6_volt_mlp_accuracy.png   MLP error       (membrane voltage)")
print("  plot7_rate_ridge_accuracy.png Ridge error     (firing rate)")
print("  plot8_rate_mlp_accuracy.png   MLP error       (firing rate)")
print("  plot9_chip_energy.png         SANA-FE energy per window")
print("  plot10_energy_comparison.png  chip vs CPU energy")
