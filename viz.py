import numpy as np
from functools import lru_cache
import time
import math
import pandas as pd
from scipy.ndimage import gaussian_filter
import random
import plotly.graph_objects as go

np.random.seed(2)
elevation_values = np.random.uniform(low=00, high=90, size=(15, 15))
sigma = 2.6
smoothed_elevation = gaussian_filter(elevation_values, sigma=sigma)
smoothed_elevation_int = np.round(smoothed_elevation).astype(int)
map_data = pd.DataFrame(smoothed_elevation_int)

def run_ql(map_data, start, goal):
    env = Environment(map_data.values, start, goal)
    agent = QLearningAgent(env)

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(env.actions[action])
            agent.update_q_table(state, action, reward, next_state)
            state = next_state

    state = start
    optimal_path = [state]
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(env.actions[action])
        optimal_path.append(next_state)
        state = next_state

    return optimal_path

def run_algorithms(num_runs=5):
    np.random.seed(2)
    elevation_values = np.random.uniform(low=00, high=90, size=(15, 15))
    sigma = 2.6
    smoothed_elevation = gaussian_filter(elevation_values, sigma=sigma)
    smoothed_elevation_int = np.round(smoothed_elevation).astype(int)
    map_data = pd.DataFrame(smoothed_elevation_int)

    algorithms = {
        "A*": Astar.run,
        "Simulated Annealing": SimulatedAnnealing.run,
        "ACO": ACO.run,
        "Q-Learning": run_ql,
        # "ACO_QL" :ACO_QL.run
        }

    results = {}
    for name, algorithm in algorithms.items():
        results[name] = {
            "runtimes": [],
            "distances": [],
            "displacements": [],
            "elevation_differences": [],
        }
        for _ in range(num_runs):
            start_time = time.time()
            start = (0, 0)
            goal = (13, 13)

            if name == "Q-Learning":
                path = algorithm(map_data, start, goal)
            else:
                path, fig = algorithm(map_data.values, start, goal)

            end_time = time.time()
            runtime = end_time - start_time

            distance = 0
            displacement = 0
            elevation_difference = 0
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                distance += math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
                elevation_difference += abs(
                    map_data.values[y2][x2] - map_data.values[y1][x1]
                )

            displacement = math.sqrt(
                ((goal[0] - start[0]) ** 2) + ((goal[1] - start[1]) ** 2)
            )

            results[name]["runtimes"].append(runtime)
            results[name]["distances"].append(distance)
            results[name]["displacements"].append(displacement)
            results[name]["elevation_differences"].append(elevation_difference)

    return results


def plot_results(results):
    line_styles = {
        "A*": {"dash": "solid", "width": 4},
        "Simulated Annealing": {"dash": "dash", "width": 4},
        "ACO": {"dash": "dot", "width": 4},
        "Q-Learning": {"dash": "dashdot", "width": 4},
    }

    fig = go.Figure()

    for name, data in results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(data["runtimes"]) + 1)),
                y=data["runtimes"],
                mode="lines+markers+text",
                name=name,
                line=dict(
                    dash=line_styles[name]["dash"],
                    width=line_styles[name]["width"],
                ),
                text=[f"{val:.2f}" for val in data["runtimes"]],
                textposition="top center",
                textfont=dict(size=12, color="black"),
            )
        )

    fig.update_layout(
        title="<b>Algorithm Runtimes</b>",
        xaxis_title="<b>Run Number</b>",
        yaxis_title="<b>Runtime (seconds)</b>",
        plot_bgcolor="white",
        font=dict(
            family="Arial Black",
            size=16,
            color="black"
        ),
        width=1400,
        height=500,
    )
    fig.show()

    fig = go.Figure()

    for name, data in results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(data["distances"]) + 1)),
                y=np.array(data["distances"]) / np.array(data["displacements"]),
                mode="lines+markers+text",
                name=name,
                line=dict(
                    dash=line_styles[name]["dash"],
                    width=line_styles[name]["width"],
                ),
                text=[f"{val:.2f}" for val in np.array(data["distances"]) / np.array(data["displacements"])],
                textposition="top center",
                textfont=dict(size=12, color="black"),
            )
        )

    fig.update_layout(
        title="<b>Algorithm Distances/Displacements</b>",
        xaxis_title="<b>Run Number</b>",
        yaxis_title="<b>Distance/Displacement</b>",
        plot_bgcolor="white",
        font=dict(
            family="Arial Black",
            size=16,
            color="black"
        ),
        width=1400,
        height=500,
    )

    fig.show()

    fig = go.Figure()

    for name, data in results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(data["displacements"]) + 1)),
                y=data["displacements"],
                mode="lines+markers+text",
                name=name,
                line=dict(
                    dash=line_styles[name]["dash"],
                    width=line_styles[name]["width"],
                ),
                text=[f"{val:.2f}" for val in data["displacements"]],
                textposition="top center",
                textfont=dict(size=12, color="black"),
            )
        )

    fig.update_layout(
        title="<b>Algorithm Displacements</b>",
        xaxis_title="<b>Run Number</b>",
        yaxis_title="<b>Displacement</b>",
        plot_bgcolor="white",
        font=dict(
            family="Arial Black",
            size=16,
            color="black"
        ),
        width=1400,
        height=500,
    )

    fig.show()

    fig = go.Figure()

    for name, data in results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(data["elevation_differences"]) + 1)),
                y=data["elevation_differences"],
                mode="lines+markers+text",
                name=name,
                line=dict(
                    dash=line_styles[name]["dash"],
                    width=line_styles[name]["width"],
                ),
                text=[f"{val:.2f}" for val in data["elevation_differences"]],
                textposition="top center",
                textfont=dict(size=12, color="black"),
            )
        )

    fig.update_layout(
        title="<b>Algorithm Elevation Differences</b>",
        xaxis_title="<b>Run Number</b>",
        yaxis_title="<b>Elevation Difference</b>",
        plot_bgcolor="white",
        font=dict(
            family="Arial Black",
            size=16,
            color="black"
        ),
        width=1400,
        height=500,
    )

    fig.show()

    print("Elevation differences average", sum(data['elevation_differences']) / len(data['elevation_differences']))
    print("Runtimes average", sum(data['runtimes']) / len(data['runtimes']))
    print("Distances average", sum(data['distances']) / len(data['distances']))


results = run_algorithms(5)
plot_results(results)
