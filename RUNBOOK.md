# RADON — Operator Runbook

> **TL;DR — To see it working right now:**
> ```bash
> # 1. Activate the environment
> .venv\Scripts\activate
> # 2. Launch the backend server
> python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
> # 3. Open your browser at:
> # http://localhost:8000
> ```

---

## 1. Prerequisites & One-Time Setup

RADON is built on Python 3.11 with PyTorch CUDA acceleration for NVIDIA GPUs (e.g. RTX 3050 Laptop GPU).

### Step 1.1: Verify Python 3.11
```bash
python --version
```
*What this does:* Verifies your installed Python version.  
*What you should see:* `Python 3.11.x`.

### Step 1.2: Create and activate virtual environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*What this does:* Creates an isolated environment and activates it.  
*What you should see:* `(.venv)` prompt prefix in your terminal.

### Step 1.3: Install dependencies
```bash
pip install -r requirements.txt
```
*What this does:* Installs PettingZoo, Gymnasium, SuperSuit, Stable-Baselines3, PyTorch CUDA 12.1, Matplotlib, FastAPI, and Uvicorn.  
*What you should see:* `Successfully installed ...` with no compilation errors.

### Step 1.4: Confirm GPU acceleration
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
*What this does:* Confirms that PyTorch detects your NVIDIA GPU.  
*What you should see:* `CUDA available: True | Device: NVIDIA GeForce RTX 3050 Laptop GPU` (or your machine's GPU).

---

## 2. Run the 3D Dashboard (The Main Interface)

The dashboard provides an interactive Three.js 3D multi-floor visualization, replay player, live held-out map generation, and adversarial fire inspection.

### Step 2.1: Start the backend server
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```
*What this does:* Loads `agents/final_rescuer.zip` and `agents/final_fire.zip` into GPU memory and serves both API endpoints and the dashboard.  
*What you should see:*
```text
[RADON Server] Loaded rescuer model from agents/final_rescuer.zip (device: cuda)
[RADON Server] Loaded adversary fire model from agents/final_fire.zip (device: cuda)
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 2.2: Open in your web browser
Navigate to:
```text
http://localhost:8000
```
*What you should see:*
1. Fullscreen dark 3D multi-floor building with extruded walls, stairs column, ground exits, and rescuers.
2. Top bar showing `RADON Multi-Agent RL 3D`, status dot green `Backend Online (Model Ready)`, and `GPU: CUDA`.
3. Left panel with live telemetry: Evacuated, Rescuers Surviving, Active Fire, Step counter, and individual rescuer badges.
4. Right panel with replay selector, fire adversary toggle (`Curriculum` vs `Adversary PPO`), and the **Generate on Unseen Map** button.
5. Bottom playback bar with Play/Pause, Step Forward/Back, scrubber timeline, and speed selectors (0.5x, 1x, 2x, 4x).

### Step 2.3: Operating the 3D controls
- **Rotate:** Left-click and drag anywhere on the canvas.
- **Pan:** Right-click and drag.
- **Zoom:** Mouse scroll wheel.
- **Floor Isolation:** In the left panel, click `Both Floors`, `Floor 0 (Exits)`, or `Floor 1 (Upper)` to isolate floors.
- **Reset Camera:** Click `Reset Camera` in the top bar or press <kbd>R</kbd>.

### Step 2.4: Generating live episodes on fresh held-out buildings
1. In the right panel, select your fire mode:
   - **Curriculum:** Standard stochastic fire propagation.
   - **Adversary PPO:** Learned fire policy targeting rescuer transit routes and stairs.
2. Click **Generate on Unseen Map** (or press <kbd>G</kbd>).
3. The server runs inference on a fresh seed ($50,000+$) and instantly loads the new building architecture into the 3D canvas with auto-playback.

---

## 3. Re-Run Policy Evaluation (100-Episode Benchmark)

To re-verify the trained model's generalization across 100 fresh, held-out building seeds:

```bash
python -m train.evaluate --model-path agents/final_rescuer.zip --preset training_day4_base --episodes 100 --base-seed 50000 --fresh-maps --replay-out replays/day5_trained.json
```
*What this does:* Evaluates both the random baseline and the trained PPO rescuer over 100 never-seen procedural buildings (`seeds 50000..50099`).  
*What you should see:*
A side-by-side terminal comparison table showing:
- Success rate (all rescued): ~11.0% (trained) vs 0.0% (random).
- Mean hostages rescued: ~0.69 / 2 (trained) vs 0.20 / 2 (random).
- Rescuer survival rate: ~75.7% (trained) vs 55.3% (random).
- Fire deaths: ~0.73 (trained) vs 1.34 (random).

---

## 4. Re-Run or Continue Training

To continue training or train a new policy from scratch:

```bash
python -m train.train_ppo --preset training_day4_base --total-timesteps 250000 --num-vec-envs 8 --checkpoint-freq 125000 --resume-path agents/final_rescuer.zip --save-path agents/final_rescuer.zip
```
*What this does:* Runs PPO with 8 vectorized parallel environments (24 parallel agents) on GPU, logging held-out test generalization to `metrics/final_rescuer.csv`.  
*What you should see:* Step progress counter, throughput (~1,800 steps/sec), and periodic evaluation logs displaying `train_success` vs `test_success`.

---

## 5. Watch an Episode in the Terminal

To inspect agent decision-making frame-by-frame with ASCII rendering in your console:

### Watch standard evacuation against curriculum fire:
```bash
python -m train.watch --model-path agents/final_rescuer.zip --preset training_day4_base --seed 50042 --delay 0.05
```

### Watch defense against the learned adversary fire:
```bash
python -m train.watch --model-path agents/final_rescuer.zip --fire-model agents/final_fire.zip --preset training_day4_base --seed 50043 --delay 0.05
```
*What this does:* Steps through an episode printing multi-floor ASCII floor maps (`F0`, `F1`), agent actions, carrying states, pickups, stair crossings, deliveries, and fire expansions.

---

## 6. Regenerate Publication Plots

To re-plot high-resolution learning curves from training logs:

```bash
python -m train.plot_curve --csv-path metrics/final_rescuer.csv --output-path metrics/day5_learning_curve.png --title "RADON — Day 5 Finale 1.65M-Step Generalization Curve"
```
*What this does:* Generates a publication-grade 3-panel PNG plot saved to `metrics/day5_learning_curve.png`.

---

## 7. Demo Video Recording Guide (Click-Sequence)

Follow this exact 90-second sequence when recording your screen demo:

1. **Setup:** Launch `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000` and open `http://localhost:8000` in full screen (F11).
2. **Scene & Playback (0:00 - 0:25):**
   - Press <kbd>Space</kbd> or click the Play button.
   - Rotate the 3D building with left-click to show both floors simultaneously.
   - Point out rescuer agents climbing stairs, picking up hostages, and carrying them down to ground exits while fire spreads.
3. **Floor Isolation (0:25 - 0:40):**
   - Click `Floor 0 (Exits)` on the left panel to show ground-floor exit corridors.
   - Click `Floor 1 (Upper)` to show upper-level search and stair access.
   - Click `Both Floors` to return to multi-floor overview.
4. **Live Generation on Unseen Map (0:40 - 1:05):**
   - In the right panel, keep `Curriculum` selected and click **Generate on Unseen Map**.
   - Show the toast message announcing the newly generated seed.
   - Watch the rescuers clear the fresh building live.
5. **Adversary Fire Attack (1:05 - 1:25):**
   - In the right panel, switch Fire Adversary Mode to **Adversary PPO**.
   - Click **Generate on Unseen Map**.
   - Point out the learned fire actively igniting chokepoints near stairs and corridors to trap rescuers.
6. **Telemetry & Curves (1:25 - 1:35):**
   - Click **Learning Curves & Benchmark** in the top bar.
   - Switch between the `1.65M-Step Learning Curve`, `Self-Play Arms Race`, and `100-Episode Benchmark` tabs.

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `[WinError 10048] address already in use` | Port 8000 is occupied by another process. | Run on another port: `python -m uvicorn server.app:app --port 8080`, then open `http://localhost:8080`. |
| `CUDA available: False` | PyTorch was installed without CUDA support or driver is missing. | Reinstall PyTorch CUDA wheel: `pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121`. |
| `HTTP 503: Rescuer model not loaded` | File `agents/final_rescuer.zip` is missing. | Confirm `agents/final_rescuer.zip` exists in the repository. If not, copy `agents/base_frozen.zip` to `agents/final_rescuer.zip`. |
| Backend dot is red `Offline (Viewing Mode)` | Uvicorn server is not running in a terminal. | Start the server with `python -m uvicorn server.app:app --port 8000`. The frontend will automatically reconnect. Pre-recorded replays remain fully playable offline. |

