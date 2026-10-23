"""Record a GIF of one episode played by any `Agent`.

By default the episode uses the first *test* seed (10000) rather than a
hand-picked one, so the GIF is a representative sample, not a highlight reel.

Usage::

    python -m lunar_rl.record --agent random --out assets/random_agent.gif
    python -m lunar_rl.record --agent sb3_ppo --model models/sb3_ppo/seed_0/best_model.zip --out assets/sb3_ppo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lunar_rl.agents import AGENT_NAMES, Agent, load_agent
from lunar_rl.envs import make_env
from lunar_rl.utils import TEST_SEED_START, get_logger

log = get_logger("record")


def _annotate(frame: np.ndarray, text: str, width: int) -> np.ndarray:
    img = Image.fromarray(frame)
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10, width // 28))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def record_episode(
    agent: Agent,
    seed: int = TEST_SEED_START,
    label: str = "",
    env_kwargs: dict[str, Any] | None = None,
    frame_skip: int = 2,
    width: int = 400,
) -> tuple[list[np.ndarray], float]:
    """Play one deterministic episode and return (annotated frames, episode return)."""
    env = make_env(render_mode="rgb_array", **(env_kwargs or {}))
    obs, _ = env.reset(seed=seed)
    frames, total, step = [], 0.0, 0
    done = False
    while True:
        if step % frame_skip == 0 or done:
            frames.append(_annotate(env.render(), f"{label} | return {total:.1f}", width))
        if done:
            break
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
        total += float(reward)
        step += 1
        done = terminated or truncated
    env.close()
    return frames, total


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 25, hold_last_s: float = 1.0) -> None:
    """Write frames as a looping GIF, holding the final frame so the outcome is visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames + [frames[-1]] * int(hold_last_s * fps)
    iio.imwrite(path, np.stack(frames), duration=1000 / fps, loop=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, choices=AGENT_NAMES)
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=TEST_SEED_START, help="Episode seed.")
    p.add_argument("--label", default=None, help="Text overlay (defaults to the agent name).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame-skip", type=int, default=2, help="Keep every n-th frame (env runs at 50 FPS).")
    p.add_argument("--width", type=int, default=400)
    p.add_argument("--wind", action="store_true")
    p.add_argument("--wind-power", type=float, default=15.0)
    p.add_argument("--turbulence-power", type=float, default=1.5)
    args = p.parse_args()

    env_kwargs = {"enable_wind": args.wind, "wind_power": args.wind_power,
                  "turbulence_power": args.turbulence_power}
    agent = load_agent(args.agent, args.model)
    frames, total = record_episode(agent, args.seed, args.label or args.agent, env_kwargs,
                                   args.frame_skip, args.width)
    save_gif(frames, args.out, fps=50 // args.frame_skip)
    size_mb = args.out.stat().st_size / 1e6
    log.info("Saved %s (%d frames, return %.1f, %.2f MB)", args.out, len(frames), total, size_mb)


if __name__ == "__main__":
    main()
