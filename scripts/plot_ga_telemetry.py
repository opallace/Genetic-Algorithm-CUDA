#!/usr/bin/env python3

import argparse
import csv
import os
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live plot for GA telemetry CSV."
    )

    parser.add_argument(
        "--csv",
        default="ga_telemetry.csv",
        help="Telemetry CSV file path."
    )

    parser.add_argument(
        "--backend",
        default="QtAgg",
        help="Matplotlib backend: QtAgg, WebAgg, TkAgg, etc."
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=200,
        help="Update interval in milliseconds."
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="Maximum number of points kept in memory."
    )

    parser.add_argument(
        "--tail-existing",
        action="store_true",
        help="Ignore existing rows and only plot new rows."
    )

    return parser.parse_args()


args = parse_args()

import matplotlib

matplotlib.use(args.backend)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


REQUIRED_COLUMNS = [
    "generation",
    "current_best_fitness",
    "global_best_fitness",
    "mutation_rate",
    "mutation_eta",
    "crossover_eta",
    "population_size",
]


class IncrementalCsvReader:
    def __init__(self, filename):
        self.filename = filename
        self.offset = 0
        self.columns = None
        self.partial_line = ""

    def skip_existing(self):
        if not os.path.exists(self.filename):
            return

        with open(self.filename, "r", encoding="utf-8", newline="") as file:
            header = file.readline()

            if header:
                self.columns = next(csv.reader([header.strip()]))

            file.seek(0, os.SEEK_END)
            self.offset = file.tell()

    def read_new_rows(self):
        if not os.path.exists(self.filename):
            return []

        current_size = os.path.getsize(self.filename)

        if current_size < self.offset:
            self.offset = 0
            self.columns = None
            self.partial_line = ""

        with open(self.filename, "r", encoding="utf-8", newline="") as file:
            file.seek(self.offset)
            chunk = file.read()
            self.offset = file.tell()

        if not chunk:
            return []

        data = self.partial_line + chunk
        lines = data.splitlines(keepends=True)

        if lines and not lines[-1].endswith("\n"):
            self.partial_line = lines.pop()
        else:
            self.partial_line = ""

        rows = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            values = next(csv.reader([line]))

            if self.columns is None:
                self.columns = values
                continue

            if values == self.columns:
                continue

            if len(values) != len(self.columns):
                continue

            row = dict(zip(self.columns, values))
            rows.append(row)

        return rows


class TelemetryBuffer:
    def __init__(self, max_points):
        self.generation = deque(maxlen=max_points)
        self.current_best_fitness = deque(maxlen=max_points)
        self.global_best_fitness = deque(maxlen=max_points)
        self.mutation_rate = deque(maxlen=max_points)
        self.mutation_eta = deque(maxlen=max_points)
        self.crossover_eta = deque(maxlen=max_points)
        self.population_size = deque(maxlen=max_points)

    def clear(self):
        self.generation.clear()
        self.current_best_fitness.clear()
        self.global_best_fitness.clear()
        self.mutation_rate.clear()
        self.mutation_eta.clear()
        self.crossover_eta.clear()
        self.population_size.clear()

    def append_row(self, row):
        for column in REQUIRED_COLUMNS:
            if column not in row:
                return False

        try:
            new_generation = int(row["generation"])

            if len(self.generation) > 0:
                last_generation = self.generation[-1]

                if new_generation <= last_generation:
                    self.clear()

            self.generation.append(new_generation)
            self.current_best_fitness.append(float(row["current_best_fitness"]))
            self.global_best_fitness.append(float(row["global_best_fitness"]))
            self.mutation_rate.append(float(row["mutation_rate"]))
            self.mutation_eta.append(float(row["mutation_eta"]))
            self.crossover_eta.append(float(row["crossover_eta"]))
            self.population_size.append(int(row["population_size"]))

            return True

        except ValueError:
            return False

    def empty(self):
        return len(self.generation) == 0

    def last(self):
        if self.empty():
            return None

        return {
            "generation": self.generation[-1],
            "current_best_fitness": self.current_best_fitness[-1],
            "global_best_fitness": self.global_best_fitness[-1],
            "mutation_rate": self.mutation_rate[-1],
            "mutation_eta": self.mutation_eta[-1],
            "crossover_eta": self.crossover_eta[-1],
            "population_size": self.population_size[-1],
        }


def format_float(value, precision=10):
    return f"{value:.{precision}g}"


reader = IncrementalCsvReader(args.csv)

if args.tail_existing:
    reader.skip_existing()

buffer = TelemetryBuffer(args.max_points)

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

fig.suptitle("GA telemetry", y=0.985)

status_text = fig.text(
    0.5,
    0.945,
    "",
    ha="center",
    va="top",
    fontsize=10,
    fontweight="bold"
)

fitness_current_line, = axs[0, 0].plot([], [], label="Current best")
fitness_global_line, = axs[0, 0].plot([], [], label="Global best")

mutation_rate_line, = axs[0, 1].plot([], [])

mutation_eta_line, = axs[1, 0].plot([], [], label="Mutation eta")
crossover_eta_line, = axs[1, 0].plot([], [], label="Crossover eta")

population_line, = axs[1, 1].plot([], [], drawstyle="steps-post")

fitness_text = axs[0, 0].text(
    0.02,
    0.02,
    "",
    transform=axs[0, 0].transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

mutation_text = axs[0, 1].text(
    0.02,
    0.95,
    "",
    transform=axs[0, 1].transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

eta_text = axs[1, 0].text(
    0.02,
    0.02,
    "",
    transform=axs[1, 0].transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

population_text = axs[1, 1].text(
    0.02,
    0.95,
    "",
    transform=axs[1, 1].transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)


def configure_axes():
    axs[0, 0].set_title("Fitness evolution")
    axs[0, 0].set_xlabel("Generation")
    axs[0, 0].set_ylabel("Fitness")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].set_title("Mutation rate")
    axs[0, 1].set_xlabel("Generation")
    axs[0, 1].set_ylabel("Mutation rate")
    axs[0, 1].grid(True)

    axs[1, 0].set_title("Eta parameters")
    axs[1, 0].set_xlabel("Generation")
    axs[1, 0].set_ylabel("Eta")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].set_title("Population size")
    axs[1, 1].set_xlabel("Generation")
    axs[1, 1].set_ylabel("Individuals")
    axs[1, 1].grid(True)


configure_axes()


def autoscale_all_axes():
    for ax in axs.flat:
        ax.relim()
        ax.autoscale_view()


def update_plot(_):
    new_rows = reader.read_new_rows()

    if not new_rows and not buffer.empty():
        return []

    for row in new_rows:
        buffer.append_row(row)

    if buffer.empty():
        return []

    generation = list(buffer.generation)

    fitness_current_line.set_data(
        generation,
        list(buffer.current_best_fitness)
    )

    fitness_global_line.set_data(
        generation,
        list(buffer.global_best_fitness)
    )

    mutation_rate_line.set_data(
        generation,
        list(buffer.mutation_rate)
    )

    mutation_eta_line.set_data(
        generation,
        list(buffer.mutation_eta)
    )

    crossover_eta_line.set_data(
        generation,
        list(buffer.crossover_eta)
    )

    population_line.set_data(
        generation,
        list(buffer.population_size)
    )

    last = buffer.last()

    status_text.set_text(
        f"Generation: {last['generation']} | "
        f"Current best: {format_float(last['current_best_fitness'], 8)} | "
        f"Global best: {format_float(last['global_best_fitness'], 8)} | "
        f"Mutation rate: {format_float(last['mutation_rate'], 8)} | "
        f"Mutation eta: {format_float(last['mutation_eta'], 6)} | "
        f"Crossover eta: {format_float(last['crossover_eta'], 6)} | "
        f"Population: {last['population_size']}"
    )

    fitness_text.set_text(
        f"generation = {last['generation']}\n"
        f"current = {format_float(last['current_best_fitness'], 8)}\n"
        f"global = {format_float(last['global_best_fitness'], 8)}"
    )

    mutation_text.set_text(
        f"mutation_rate = {format_float(last['mutation_rate'], 8)}"
    )

    eta_text.set_text(
        f"mutation_eta = {format_float(last['mutation_eta'], 8)}\n"
        f"crossover_eta = {format_float(last['crossover_eta'], 8)}"
    )

    population_text.set_text(
        f"population = {last['population_size']}"
    )

    autoscale_all_axes()

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    return [
        fitness_current_line,
        fitness_global_line,
        mutation_rate_line,
        mutation_eta_line,
        crossover_eta_line,
        population_line,
    ]


animation = FuncAnimation(
    fig,
    update_plot,
    interval=args.interval,
    cache_frame_data=False
)

plt.show()