import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle, Circle

def read_data(filename):
    data = pd.read_csv(
        filename,
        sep=";",
        decimal=","
    )

    time = data["time"].to_numpy()
    x = data["x"].to_numpy()
    theta = data["theta"].to_numpy()

    return time, x, theta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV com colunas: time,x,theta")
    parser.add_argument("--length", type=float, default=1.0, help="Comprimento do pêndulo")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="suspended_load_animation.gif")
    parser.add_argument("--stride", type=int, default=1, help="Pula frames para deixar a animação mais leve")

    args = parser.parse_args()

    time, x, theta = read_data(args.csv_file)

    indices = np.arange(0, len(time), args.stride)

    l = args.length

    # theta = 0 significa pêndulo vertical para baixo
    pivot_x = x
    pivot_y = np.zeros_like(x)

    bob_x = pivot_x + l * np.sin(theta)
    bob_y = pivot_y - l * np.cos(theta)

    cart_width = 0.30
    cart_height = 0.16
    wheel_radius = 0.035
    bob_radius = 0.06

    x_min = min(np.min(x), np.min(bob_x)) - 0.5
    x_max = max(np.max(x), np.max(bob_x)) + 0.5

    y_min = -1.25 * l
    y_max = 0.35 * l

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Vertical position (m)")
    ax.set_title("Suspended load motion")

    # Trilho
    ax.plot([x_min, x_max], [0, 0], linewidth=1)

    # Carrinho
    cart = Rectangle(
        (x[0] - cart_width / 2, -cart_height / 2),
        cart_width,
        cart_height,
        fill=False,
        linewidth=2
    )
    ax.add_patch(cart)

    # Rodas
    wheel_left = Circle(
        (x[0] - cart_width * 0.3, -cart_height / 2 - wheel_radius),
        wheel_radius,
        fill=False,
        linewidth=1.5
    )
    wheel_right = Circle(
        (x[0] + cart_width * 0.3, -cart_height / 2 - wheel_radius),
        wheel_radius,
        fill=False,
        linewidth=1.5
    )

    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)

    # Haste do pêndulo
    rod, = ax.plot(
        [pivot_x[0], bob_x[0]],
        [pivot_y[0], bob_y[0]],
        linewidth=2
    )

    # Massa suspensa
    bob = Circle(
        (bob_x[0], bob_y[0]),
        bob_radius
    )
    ax.add_patch(bob)

    # Pivô
    pivot, = ax.plot(
        [pivot_x[0]],
        [pivot_y[0]],
        marker="o"
    )

    info_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    def update(frame_id):
        i = indices[frame_id]

        # Atualiza carrinho
        cart.set_xy((
            x[i] - cart_width / 2,
            -cart_height / 2
        ))

        # Atualiza rodas
        wheel_left.center = (
            x[i] - cart_width * 0.3,
            -cart_height / 2 - wheel_radius
        )

        wheel_right.center = (
            x[i] + cart_width * 0.3,
            -cart_height / 2 - wheel_radius
        )

        # Atualiza haste
        rod.set_data(
            [pivot_x[i], bob_x[i]],
            [pivot_y[i], bob_y[i]]
        )

        # Atualiza massa
        bob.center = (
            bob_x[i],
            bob_y[i]
        )

        # Atualiza pivô
        pivot.set_data(
            [pivot_x[i]],
            [pivot_y[i]]
        )

        info_text.set_text(
            f"t = {time[i]:.2f} s\n"
            f"x = {x[i]:.3f} m\n"
            f"theta = {theta[i]:.4f} rad"
        )

        return cart, wheel_left, wheel_right, rod, bob, pivot, info_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000 / args.fps,
        blit=True
    )

    if args.output.endswith(".mp4"):
        writer = FFMpegWriter(fps=args.fps)
    else:
        writer = PillowWriter(fps=args.fps)

    animation.save(args.output, writer=writer)

    print(f"Animation saved as: {args.output}")


if __name__ == "__main__":
    main()