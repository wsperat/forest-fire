# examples/python/quickstart_mean_tree.py
import numpy as np
from forestfire import TargetMeanTree


def main() -> None:
    rng = np.random.default_rng(42)
    n, d = 8, 3

    # Fake features (ignored by the mean model) and a target
    X = rng.normal(size=(n, d))
    y = np.array([10.0, 12.0, 14.0, 20.0, 8.0, 13.0, 9.0, 11.0], dtype=float)

    model = TargetMeanTree.fit(X, y)
    preds = model.predict(X)

    mse = float(np.mean((preds - y) ** 2))
    print(f"n={n}, d={d}")
    print(f"mean_ = {model.mean_:.6f}")
    print("preds =", np.round(preds, 6).tolist())
    print(f"MSE   = {mse:.6f}")


if __name__ == "__main__":
    main()
