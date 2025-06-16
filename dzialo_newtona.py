import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk

def simulate_trajectory(v0, angle_rad):
    G = 6.67430e-11
    M = 5.972e24
    R = 6371000
    h = 500000
    dt = 0.2
    max_steps = 500000

    x0, y0 = 0, R + h
    x, y = x0, y0
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)

    xs, ys = [], []

    theta0 = np.arctan2(y, x)
    prev_theta = theta0
    angle_travelled = 0
    orbit_closed = False

    for step in range(max_steps):
        r = np.sqrt(x ** 2 + y ** 2)
        if r <= R:
            break  # uderzenie

        def acc(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            a = -G * M / r ** 3
            return a * x, a * y

        # RK4
        ax1, ay1 = acc(x, y)
        k1vx, k1vy = ax1 * dt, ay1 * dt
        k1x, k1y = vx * dt, vy * dt

        ax2, ay2 = acc(x + 0.5 * k1x, y + 0.5 * k1y)
        k2vx, k2vy = ax2 * dt, ay2 * dt
        k2x, k2y = (vx + 0.5 * k1vx) * dt, (vy + 0.5 * k1vy) * dt

        ax3, ay3 = acc(x + 0.5 * k2x, y + 0.5 * k2y)
        k3vx, k3vy = ax3 * dt, ay3 * dt
        k3x, k3y = (vx + 0.5 * k2vx) * dt, (vy + 0.5 * k2vy) * dt

        ax4, ay4 = acc(x + k3x, y + k3y)
        k4vx, k4vy = ax4 * dt, ay4 * dt
        k4x, k4y = (vx + k3vx) * dt, (vy + k3vy) * dt

        vx += (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
        vy += (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6
        x += (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y += (k1y + 2 * k2y + 2 * k3y + k4y) / 6

        xs.append(x)
        ys.append(y)

        if not orbit_closed and step > 100:
            dist_to_start = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            vel_dir = np.array([vx, vy])
            initial_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])

            if dist_to_start < 10000 and np.dot(vel_dir, initial_dir) > 0.95:
                orbit_closed = True
                xs.append(x0)
                ys.append(y0)
                break

    return xs, ys, orbit_closed


def start_simulation():
    v0 = speed_slider.get()
    angle_deg = angle_slider.get()
    angle_rad = np.radians(angle_deg)

    xs, ys, orbit_closed = simulate_trajectory(v0, angle_rad)

    fps = 60
    delay_frames = int(2 * fps)
    draw_frames = int(3 * fps)
    interval_ms = 1000 / fps
    total_frames = delay_frames + draw_frames

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    R = 6371000
    planet = plt.Circle((0, 0), R, color='black', fill=True)
    ax.add_patch(planet)

    buffer = R * 0.5
    ax.set_xlim(min(xs) - buffer, max(xs) + buffer)
    ax.set_ylim(min(ys) - buffer, max(ys) + buffer)

    full_traj_line, = ax.plot([], [], 'gray', linewidth=1)
    trail_line, = ax.plot([], [], 'g-', linewidth=2)
    moving_point, = ax.plot([], [], 'ro', markersize=6)

    if orbit_closed:
        ax.set_title("Armata Newtona - Orbita zamknięta (elipsa)")
    else:
        ax.set_title("Armata Newtona - Tor lotu (orbita otwarta)")

    def init():
        full_traj_line.set_data([], [])
        trail_line.set_data([], [])
        moving_point.set_data([], [])
        return full_traj_line, trail_line, moving_point

    def update(frame):
        if frame < draw_frames:
            i = int(frame / draw_frames * len(xs))
            full_traj_line.set_data([], [])
            if i < len(xs):
                trail_line.set_data(xs[:i + 1], ys[:i + 1])
                moving_point.set_data([xs[i]], [ys[i]])
            else:
                trail_line.set_data(xs, ys)
                moving_point.set_data([], [])
        else:
            full_traj_line.set_data(xs, ys)
            trail_line.set_data([], [])
            moving_point.set_data([], [])
        return full_traj_line, trail_line, moving_point

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        init_func=init,
        interval=interval_ms,
        blit=True
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.show()


root = tk.Tk()
root.title("Działo Newtona")

tk.Label(root, text="Prędkość początkowa (m/s)").pack()
speed_slider = tk.Scale(root, from_=1000, to=12000, orient="horizontal", resolution=100, length=300)
speed_slider.set(7800)
speed_slider.pack()

tk.Label(root, text="Kąt wystrzału (°)").pack()
angle_slider = tk.Scale(root, from_=0, to=90, orient="horizontal", resolution=1, length=300)
angle_slider.set(0)
angle_slider.pack()

start_button = ttk.Button(root, text="Start symulacji", command=start_simulation)
start_button.pack(pady=10)

root.mainloop()
