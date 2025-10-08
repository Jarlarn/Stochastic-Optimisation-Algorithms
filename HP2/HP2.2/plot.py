import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def main():
    # Grid
    x = np.linspace(-6, 6, 600)
    y = np.linspace(-6, 6, 600)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau(X, Y)

    # 2D contour plot
    plt.figure(figsize=(7, 6))
    levels = np.logspace(0, 5, 35, base=10)
    cs = plt.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    plt.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.6)
    plt.colorbar(label="f(x1, x2)")
    plt.title("Himmelblau Function Contours")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Mark known minima
    minima = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]
    for a, b in minima:
        plt.plot(a, b, "r*", markersize=10)
        plt.text(
            a + 0.15,
            b + 0.15,
            f"({a:.3f}, {b:.3f})",
            fontsize=12,
            color="red",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0.3),
        )

    plt.tight_layout()
    plt.savefig("HP2/HP2.2/contour.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
