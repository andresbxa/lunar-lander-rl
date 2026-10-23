import numpy as np
import pandas as pd

from lunar_rl.plot_curves import load_curves, plot


def _write_curves(root, algo):
    d = root / algo
    d.mkdir(parents=True)
    pd.DataFrame({"timesteps": [0, 100, 200], "mean_return": [-200.0, 0.0, 100.0]}).to_csv(d / "seed_0.csv")
    # Different eval points and a longer run: must be interpolated + truncated.
    pd.DataFrame({"timesteps": [0, 50, 150, 300], "mean_return": [-100.0, -50.0, 50.0, 300.0]}).to_csv(
        d / "seed_1.csv")


def test_load_curves_aligns_seeds(tmp_path):
    _write_curves(tmp_path, "algo")
    grid, returns = load_curves(tmp_path, "algo")
    np.testing.assert_array_equal(grid, [0, 100, 200])
    assert returns.shape == (2, 3)
    np.testing.assert_allclose(returns[1], [-100.0, 0.0, 50.0 + 250.0 / 3])


def test_plot_writes_png(tmp_path):
    _write_curves(tmp_path, "algo")
    out = tmp_path / "fig.png"
    plot(["algo"], tmp_path, out, random_csv=None)
    assert out.stat().st_size > 0
