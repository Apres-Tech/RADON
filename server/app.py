"""FastAPI backend application for RADON multi-agent simulation."""

import csv
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import numpy as np
from stable_baselines3 import PPO

from env.config import EnvConfig
from env.fire_controller import CurriculumFireController, PolicyFireController
from env.replay import EpisodeRecorder
from env.rescue_env import RadonRescueEnv
from env.tiles import Action

BASE_DIR = Path(__file__).resolve().parent.parent
REPLAYS_DIR = BASE_DIR / "replays"
METRICS_DIR = BASE_DIR / "metrics"
AGENTS_DIR = BASE_DIR / "agents"
DASHBOARD_DIR = BASE_DIR / "dashboard"

app = FastAPI(
    title="RADON Simulation API",
    description="Multi-agent reinforcement learning rescue simulation backend & replay server.",
    version="1.0.0",
)

# Enable CORS for local development and dashboard embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
models_state: Dict[str, Any] = {
    "rescuer_model": None,
    "rescuer_path": None,
    "fire_model": None,
    "fire_path": None,
    "device": "cpu",
    "load_error": None,
}


def load_models_at_startup() -> None:
    """Attempts to load trained rescuer and adversary fire policies once at startup."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_state["device"] = device

    # 1. Rescuer policy
    rescuer_candidates = [
        AGENTS_DIR / "showcase_rescuer.zip",
        AGENTS_DIR / "final_rescuer.zip",
        AGENTS_DIR / "rescuer_best_heldout.zip",
        AGENTS_DIR / "base_frozen.zip",
        AGENTS_DIR / "ppo_day3_two_floors.zip",
    ]
    rescuer_loaded = False
    for path in rescuer_candidates:
        if path.exists():
            try:
                models_state["rescuer_model"] = PPO.load(str(path), device=device)
                models_state["rescuer_path"] = str(path.relative_to(BASE_DIR))
                print(f"[RADON Server] Loaded rescuer model from {path} (device: {device})")
                rescuer_loaded = True
                break
            except Exception as e:
                print(f"[RADON Server] Warning: failed to load rescuer from {path}: {e}")

    if not rescuer_loaded:
        models_state["load_error"] = "No valid rescuer model found in agents/."

    # 2. Fire adversary policy
    fire_candidates = [
        AGENTS_DIR / "final_fire.zip",
        AGENTS_DIR / "fire_adversary_r1.zip",
    ]
    for path in fire_candidates:
        if path.exists():
            try:
                models_state["fire_model"] = PPO.load(str(path), device=device)
                models_state["fire_path"] = str(path.relative_to(BASE_DIR))
                print(f"[RADON Server] Loaded adversary fire model from {path} (device: {device})")
                break
            except Exception as e:
                print(f"[RADON Server] Warning: failed to load fire adversary from {path}: {e}")


@app.on_event("startup")
def startup_event():
    load_models_at_startup()


class GenerateRequest(BaseModel):
    fire_mode: str = Field(default="curriculum", description="Fire mode: 'none', 'curriculum', or 'adversarial'")
    seed: Optional[int] = Field(default=None, description="Held-out map seed (defaults to random 50000..99999)")
    preset: str = Field(default="showcase", description="Environment preset")
    max_steps: Optional[int] = Field(default=350, description="Max episode steps")


@app.get("/health")
def health():
    """Health check reporting system and model loading status."""
    return {
        "status": "healthy",
        "rescuer_loaded": models_state["rescuer_model"] is not None,
        "rescuer_model": models_state["rescuer_path"],
        "fire_loaded": models_state["fire_model"] is not None,
        "fire_model": models_state["fire_path"],
        "device": models_state["device"],
        "load_error": models_state["load_error"],
    }


@app.get("/replays")
def list_replays() -> List[Dict[str, Any]]:
    """Lists all available pre-recorded replay files with summary metadata."""
    replays: List[Dict[str, Any]] = []
    if not REPLAYS_DIR.exists():
        return replays

    for file_path in sorted(REPLAYS_DIR.glob("*.json")):
        size_kb = round(file_path.stat().st_size / 1024.0, 1)
        replay_info = {
            "name": file_path.name,
            "filename": file_path.name,
            "size_kb": size_kb,
            "modified": file_path.stat().st_mtime,
        }

        # Attempt to read quick metadata from JSON header
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = data.get("config", {})
            frames = data.get("frames", [])
            replay_info["floors"] = config.get("floors", 2)
            replay_info["grid_h"] = config.get("grid_h", 10)
            replay_info["grid_w"] = config.get("grid_w", 10)
            replay_info["n_rescuers"] = config.get("n_rescuers", 3)
            replay_info["n_hostages"] = config.get("n_hostages", 2)
            replay_info["frames_count"] = len(frames)
            replay_info["seed"] = config.get("seed")

            # Final frame stats if available
            if frames:
                last_frame = frames[-1]
                hostages = last_frame.get("hostages", [])
                rescued_count = sum(1 for h in hostages if len(h) > 3 and h[3])
                surviving_count = sum(1 for r in last_frame.get("rescuers", []) if r.get("alive", False))
                replay_info["hostages_rescued"] = rescued_count
                replay_info["rescuers_surviving"] = surviving_count
                replay_info["all_rescued"] = (rescued_count == config.get("n_hostages", 2))
        except Exception:
            pass

        replays.append(replay_info)

    # Put day5 and day4 replays at top
    replays.sort(key=lambda r: ("day5" not in r["name"], "day4" not in r["name"], r["name"]))
    return replays


@app.get("/replay/{name}")
def get_replay(name: str):
    """Retrieves a single pre-recorded replay JSON file."""
    # Basic path safety
    safe_name = Path(name).name
    if not safe_name.endswith(".json"):
        safe_name = f"{safe_name}.json"

    file_path = REPLAYS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay '{safe_name}' not found. Available: {[f.name for f in REPLAYS_DIR.glob('*.json')]}",
        )

    return FileResponse(file_path, media_type="application/json")


@app.get("/metrics")
def get_metrics():
    """Serves training learning curves and arms race progression metrics as structured JSON."""
    metrics_data: Dict[str, Any] = {
        "final_rescuer": [],
        "selfplay_arms_race": [],
        "summary": {},
    }

    final_csv = METRICS_DIR / "final_rescuer.csv"
    if final_csv.exists():
        try:
            with open(final_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for r in rows:
                    parsed_row = {}
                    for k, v in r.items():
                        try:
                            parsed_row[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            parsed_row[k] = v
                    metrics_data["final_rescuer"].append(parsed_row)

                if metrics_data["final_rescuer"]:
                    peak_test_success = max(
                        row.get("test_success", 0.0) for row in metrics_data["final_rescuer"]
                    )
                    metrics_data["summary"]["peak_test_success"] = peak_test_success
                    metrics_data["summary"]["final_timesteps"] = metrics_data["final_rescuer"][-1].get("timestep", 0)
        except Exception as e:
            metrics_data["error_rescuer"] = str(e)

    selfplay_csv = METRICS_DIR / "day4_selfplay.csv"
    if selfplay_csv.exists():
        try:
            with open(selfplay_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    parsed_row = {}
                    for k, v in r.items():
                        try:
                            parsed_row[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            parsed_row[k] = v
                    metrics_data["selfplay_arms_race"].append(parsed_row)
        except Exception as e:
            metrics_data["error_selfplay"] = str(e)

    # Provide URLs for static plots
    metrics_data["charts"] = {
        "learning_curve_png": "/metrics_static/day5_learning_curve.png",
        "arms_race_png": "/metrics_static/day4_selfplay_arms_race.png",
    }

    return metrics_data


@app.post("/generate")
def generate_episode(req: GenerateRequest):
    """Generates an episode live on a fresh unseen procedural map using trained models."""
    model = models_state["rescuer_model"]
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Rescuer model is not loaded. Ensure agents/final_rescuer.zip exists. "
                "You can still inspect pre-recorded runs in the dashboard dropdown."
            ),
        )

    # Pick held-out seed (range 50000+)
    seed = req.seed if req.seed is not None else random.randint(50000, 99999)

    # Config preset
    if req.preset == "showcase":
        config = EnvConfig.showcase(seed=seed)
    elif req.preset == "showcase_hard":
        config = EnvConfig.showcase_hard(seed=seed)
    elif req.preset == "training_day3":
        config = EnvConfig.training_day3(seed=seed)
    elif req.preset == "sanity":
        config = EnvConfig.sanity(seed=seed)
    else:
        config = EnvConfig.training_day4_base(seed=seed)

    if req.max_steps:
        config.max_steps = req.max_steps

    # Fire mode selection
    fire_mode = req.fire_mode.lower()
    fire_controller = None

    if fire_mode == "none":
        config.fire_enabled = False
    elif fire_mode == "adversarial":
        fire_model = models_state["fire_model"]
        if fire_model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Adversary fire model not loaded. Check agents/final_fire.zip or use 'curriculum' fire mode.",
            )
        fire_controller = PolicyFireController(model=fire_model, deterministic=False)
    else:  # curriculum (default)
        fire_controller = CurriculumFireController()

    try:
        env = RadonRescueEnv(config=config, fire_controller=fire_controller)
        obs, infos = env.reset(seed=seed)
        recorder = EpisodeRecorder(env)
        recorder.record_initial_frame()

        step_count = 0
        total_rewards = {a: 0.0 for a in env.possible_agents}

        while env.agents and step_count < config.max_steps:
            step_count += 1
            actions = {}
            for agent_id in env.agents:
                agent_obs = {
                    k: np.expand_dims(v, axis=0) for k, v in obs[agent_id].items()
                }
                action, _ = model.predict(agent_obs, deterministic=False)
                actions[agent_id] = int(action[0])

            obs, rewards, terminations, truncations, infos = env.step(actions)
            for a, r in rewards.items():
                total_rewards[a] += r

            recorder.record_step(rewards)

        # Outcome summary & WIN/LOSE verdict
        rescued_count = sum(1 for h in env.hostages if h.rescued)
        survivors_count = sum(1 for r in env.rescuers.values() if r.alive)
        fire_deaths = config.n_rescuers - survivors_count
        all_evacuated = (rescued_count == config.n_hostages)

        if all_evacuated:
            verdict = "WIN"
            verdict_reason = f"All {rescued_count}/{config.n_hostages} hostages safely evacuated in {step_count} steps!"
        elif survivors_count == 0:
            verdict = "LOSE"
            verdict_reason = f"Rescuer squad wiped out by fire ({rescued_count}/{config.n_hostages} saved)."
        else:
            verdict = "LOSE"
            verdict_reason = f"Time expired at step {step_count} ({rescued_count}/{config.n_hostages} saved, {survivors_count}/{config.n_rescuers} survivors)."

        replay_payload = recorder.to_dict()
        replay_payload["summary"] = {
            "seed": seed,
            "fire_mode": fire_mode,
            "steps": step_count,
            "hostages_rescued": rescued_count,
            "total_hostages": config.n_hostages,
            "rescuers_surviving": survivors_count,
            "total_rescuers": config.n_rescuers,
            "fire_deaths": fire_deaths,
            "all_rescued": all_evacuated,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "team_return": round(sum(total_rewards.values()), 2),
            "model_used": models_state["rescuer_path"],
            "fire_model_used": models_state["fire_path"] if fire_mode == "adversarial" else None,
        }

        return replay_payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Episode generation failed: {str(e)}",
        )


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🔥</text></svg>"""


@app.get("/favicon.ico")
def favicon():
    """Serves an inline flame SVG favicon to avoid 404 errors."""
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/style.css")
def get_style():
    """Serves the dashboard stylesheet directly at /style.css."""
    css_path = DASHBOARD_DIR / "style.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js")
def get_app_js():
    """Serves the dashboard script directly at /app.js."""
    js_path = DASHBOARD_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


# Static assets
if METRICS_DIR.exists():
    app.mount("/metrics_static", StaticFiles(directory=str(METRICS_DIR)), name="metrics_static")

if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
    # Also mount at root as fallback for any other dashboard static assets
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="root_static")


@app.get("/")
def index():
    """Serves the 3D dashboard directly at the root URL."""
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse(
        "<h2>RADON Simulation API is running.</h2><p>Dashboard not found at <code>dashboard/index.html</code>.</p>"
    )

