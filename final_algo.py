import numpy as np
from functools import lru_cache
import time
import math
import pandas as pd
from scipy.ndimage import gaussian_filter
import random
import plotly.graph_objects as go


'''<------------------------------------------------------------------------------------------------------------------------------->'''


class Astar:

    def __init__(self, matrix):
        self.mat = self.prepare_matrix(matrix)

    class Node:
        def __init__(self, x, y, weight=0):
            self.x = x
            self.y = y
            self.weight = weight
            self.cost=0
            self.parent = None

    def print(self):
        for y in self.mat:
            print(y)

    def prepare_matrix(self, mat):
        matrix_for_astar = []
        for y, line in enumerate(mat):
            tmp_line = []
            for x, weight in enumerate(line):
                tmp_line.append(self.Node(x, y, weight=weight))
            matrix_for_astar.append(tmp_line)
        return matrix_for_astar

    def print_weight_matrix(self):
      for row in self.mat:
          row_weights = [node.weight for node in row]
          print(row_weights)

    def equal(self, current, end):#This method checks if two nodes have the same coordinates.
        return current.x == end.x and current.y == end.y

    def manhattan(self, current, other):
        return (abs(current.x - other.x) + abs(current.y - other.y))# |(X1-x2)|+|(y1-y2)|

    def neighbours(self, matrix, current):# check karta hai ki corner pe hai toh left or upper nahi jaa sakta hai, null ya unknown values me nahi jata
        neighbours_list = []
        if current.x - 1 >= 0 and matrix[current.y][current.x - 1].weight is not None:
            neighbours_list.append(matrix[current.y][current.x - 1])
        if current.y - 1 >= 0 and matrix[current.y - 1][current.x].weight is not None:
            neighbours_list.append(matrix[current.y - 1][current.x])
        if current.y + 1 < len(matrix) and matrix[current.y + 1][current.x].weight is not None:
            neighbours_list.append(matrix[current.y + 1][current.x])
        if current.x + 1 < len(matrix[0]) and matrix[current.y][current.x + 1].weight is not None:
            neighbours_list.append(matrix[current.y][current.x + 1])
        return neighbours_list

    def reconstruct_path(self, end):#This method reconstructs the optimal path from the start node to the end node.
        node_tmp = end
        path = []
        while (node_tmp):
            path.append([node_tmp.x, node_tmp.y])
            node_tmp = node_tmp.parent
        return list(reversed(path))

    def main(self, point_start, point_end):
        matrix = self.mat

        start = self.Node(point_start[0], point_start[1])
        end = self.Node(point_end[0], point_end[1])
        closed_list = []
        open_list = [start]

        def sort_key(node):
            return node.cost

        while open_list:
            open_list.sort(key=sort_key)
            current_node =  open_list.pop(0)

            for node in open_list:
                if node.cost < current_node.cost and node not in closed_list:
                    current_node = node

            if self.equal(current_node, end):
                return self.reconstruct_path(current_node)

            for node in open_list:
                if self.equal(current_node, node):
                    open_list.remove(node)
                    break

            closed_list.append(current_node)

            for neighbour in self.neighbours(matrix, current_node):
                if neighbour in closed_list:
                    continue
                if (neighbour.cost < current_node.cost) or (neighbour not in open_list):
                    neighbour.parent = current_node
                    neighbour.cost = 2 *self.calculate_parents_weight(neighbour) + self.manhattan(neighbour, end)

                if neighbour not in open_list:
                    open_list.append(neighbour)

        return None

    def calculate_parents_weight(self, node):#This method calculates the sum of weights of nodes in the path from a given node to the start node.
        weight_sum = node.weight
        while node.parent is not None:
            weight_sum += node.parent.weight
            node= node.parent
        return weight_sum

    # def map_projection(self, map, result):#This method projects the map with the optimal path highlighted for visualization.
    #   for i, row in enumerate(map):
    #       for j, item in enumerate(row):
    #           if item is None:
    #               map[i][j] = np.nan

    # #   sns.set_theme()
    #   plt.figure(figsize=[10,10])
    #   ax = sns.heatmap(map, linewidths=0.5)
    #   result_df = pd.DataFrame(result, columns=['x', 'y'])
    #   plt.plot(result_df.x + 0.5, result_df.y + 0.5, linewidth=10)
    #   plt.show()

    def maps(self, map2, map):
        for n in range(len(map)):
            for k in range(len(map[0])):
                if map2[n][k] is not None:
                    map2[n][k] += 10
                else:
                    map2[n][k] = None

        for n in range(len(map)):
            for k in range(len(map[0])):
                if map2[n][k] is not None and map[n][k] is not None:
                    map2[n][k] += map[n][k]
                else:
                    map2[n][k] = None

    def run(map_data, src, dest):
        astar = Astar(map_data)
        result = astar.main(src, dest)

        fig = go.Figure(go.Scatter(x=[src[0]], y=[src[1]], mode='markers', marker_color='red', name='Start'))
        fig.add_trace(go.Scatter(x=[dest[0]], y=[dest[1]], mode='markers', marker_color='green', name='End'))

        path_x, path_y = zip(*result)
        fig.add_trace(go.Scatter(x=path_y, y=path_x, mode='lines', line_color='red', name='Path'))

        fig.update_layout(
            title='Optimal Path(A*)',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
        )

        earth_elevation_colorscale = [
            [0.0, 'rgb(0, 0, 128)'],
            [0.1, 'rgb(0, 128, 255)'],
            [0.2, 'rgb(102, 204, 0)'],
            [0.4, 'rgb(0, 102, 0)'],
            [0.6, 'rgb(153, 153, 0)'],
            [0.8, 'rgb(153, 102, 51)'],
            [1.0, 'rgb(255, 255, 255)']
        ]


        fig.add_trace(go.Heatmap(z=map_data, x0=0, y0=0, dx=1, dy=1, colorscale=earth_elevation_colorscale))

        return result, fig


'''<------------------------------------------------------------------------------------------------------------------------------->'''


class SimulatedAnnealing:
    def __init__(self, heatmap_data, start_point, end_point, initial_temperature=10000, cooling_rate=0.0001, num_iterations=1000):
        self.heatmap_data = heatmap_data
        self.start_point = start_point
        self.end_point = end_point
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations

    def calculate_path_cost(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]
            elevation_diff = abs(self.heatmap_data[next_point] - self.heatmap_data[current_point])
            total_cost += elevation_diff
        return total_cost

    def acceptance_probability(self, current_cost, new_cost, temperature):
        if new_cost < current_cost:
            return 1
        else:
            return math.exp((current_cost - new_cost) / temperature)

    def generate_initial_solution(self):
        current_solution = [self.start_point]
        while current_solution[-1] != self.end_point:
            next_point = self.get_random_neighbor(current_solution[-1])
            current_solution.append(next_point)
        return current_solution

    def get_random_neighbor(self, point):
        # Generate random neighbor within 8-neighborhood
        row, col = point
        neighbor_row = random.choice([row-1, row, row+1])
        neighbor_col = random.choice([col-1, col, col+1])
        neighbor_row = max(0, min(neighbor_row, self.heatmap_data.shape[0] - 1))
        neighbor_col = max(0, min(neighbor_col, self.heatmap_data.shape[1] - 1))
        return (neighbor_row, neighbor_col)

    def main(self):
        current_solution = self.generate_initial_solution()
        current_cost = self.calculate_path_cost(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        temperature = self.initial_temperature
        for _ in range(self.num_iterations):
            new_solution = current_solution.copy()

            perturb_index = random.randint(1, len(new_solution) - 2)
            new_solution[perturb_index] = self.get_random_neighbor(new_solution[perturb_index - 1])

            new_cost = self.calculate_path_cost(new_solution)

            if self.acceptance_probability(current_cost, new_cost, temperature) > random.random():
                current_solution = new_solution
                current_cost = new_cost

            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

            temperature *= self.cooling_rate

        return best_solution, best_cost

    def run(map_data, src, dest):
        sa = SimulatedAnnealing(map_data, src, dest)
        best_solution, best_cost = sa.main()
        res = best_solution
        seen = {}
        for i,j in enumerate(res):
            end = len(res)-1
            for k in range(end,i,-1):
                if(j == res[k]):
                    del res[i:k]
                    break

        fig = go.Figure(go.Scatter(x=[src[0]], y=[src[1]], mode='markers', marker_color='red', name='Start'))
        fig.add_trace(go.Scatter(x=[dest[0]], y=[dest[1]], mode='markers', marker_color='green', name='End'))

        path_x, path_y = zip(*res)
        fig.add_trace(go.Scatter(x=path_y, y=path_x, mode='lines', line_color='red', name='Path'))

        fig.update_layout(
            title='Optimal Path(SA)',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
        )

        earth_elevation_colorscale = [
            [0.0, 'rgb(0, 0, 128)'],
            [0.1, 'rgb(0, 128, 255)'],
            [0.2, 'rgb(102, 204, 0)'],
            [0.4, 'rgb(0, 102, 0)'],
            [0.6, 'rgb(153, 153, 0)'],
            [0.8, 'rgb(153, 102, 51)'],
            [1.0, 'rgb(255, 255, 255)']
        ]


        fig.add_trace(go.Heatmap(z=map_data, x0=0, y0=0, dx=1, dy=1, colorscale=earth_elevation_colorscale))

        return res, fig

'''<------------------------------------------------------------------------------------------------------------------------------->'''
class ACO():

    class Ant():
        def __init__(self, start_coords, end_coords):
            self.curr = start_coords
            self.start_pos = start_coords
            self.final_pos = end_coords
            self.final_node = False
            self.visited_nodes = []
            self.mark_visited(self.curr)

        def setup(self):
            self.visited_nodes = [self.start_pos]
            # print("After: ",self.visited_nodes)
            self.curr = self.start_pos

        def mark_visited(self, coords):
            self.visited_nodes.append(coords)

        def move_ant(self, coords):
            self.curr  = coords
            self.mark_visited(coords)

        def check_final(self, coords):
            if coords == self.final_pos:
                self.final_node = True

        def reset(self):
            self.visited_nodes = []
            self.final_node = False

    def __init__(self, map, start_coords, end_coords, iters = 10, ants = 50, p = 0.5, early_stop = 0, alpha=1, beta=1, gamma=1):
        self.map = map
        self.iters = iters
        self.no_ants = ants
        self.p = p
        self.early_stop = early_stop
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.paths = []
        self.best_path = []
        self.pheromone = np.ones((len(self.map[0]), len(self.map)))
        self.ants = []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def init_ants(self):
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.start_coords, self.end_coords))
        return ants

    def get_neighbours(self, curr):
        neighbours = []
        x_list = [1,0,-1] if curr[0] != 0 else [1,0]
        y_list = [1,0,-1] if curr[1] != 0 else [1,0]

        for dx in x_list:
            for dy in y_list:
                if ((curr[0] + dx) < (len(self.map[0]) - 1)) and ((curr[1] + dy) < (len(self.map) - 1)) and (curr[0] + dx) >= 0 and (curr[1] + dy) >= 0:
                    neighbours.append((curr[0] + dx, curr[1] + dy))
        neighbours.remove(curr)
        return neighbours

    def select_new_node(self, curr, ant):
        total_sum = 0
        neighbours = self.get_neighbours(curr)

        epsilon = 1e-6
        attractiveness = []
        for n in neighbours:
            pheromone = self.pheromone[n[0]][n[1]]
            distance = math.sqrt(((n[0] - self.end_coords[0])**2) + ((n[1] - self.end_coords[1])**2)) + 0.1
            elevation_diff = abs(self.map[curr[0]][curr[1]] - self.map[n[0]][n[1]])
            attr = (pheromone ** self.alpha) * ((1 / distance) ** self.beta) * ((1 / (elevation_diff + 1 + epsilon)) ** self.gamma)
            attractiveness.append(attr)

        total_sum = sum(attractiveness)
        prob = [a / total_sum for a in attractiveness]
        new = np.random.choice((range(len(neighbours))), p=prob)
        return neighbours[new]

    def update_weights(self):
        self.paths.sort(key=len)
        self.best_path = self.paths[0]
        for i in list(self.best_path)[:-1]:
            x_coord, y_coord = i
            self.pheromone[x_coord][y_coord] += (1 - self.p) * self.pheromone[x_coord][y_coord]

    def get_repeats(self, path, element):
        res = []
        offset = -1
        while True:
            try:
                offset = path.index(element, offset+1)
            except ValueError:
                return res
            res.append(offset)

    def remove_loops(self, path):
        path = path
        for element in path:
            repeats = self.get_repeats(path, element)
            repeats.reverse()
            for i, hit in enumerate(repeats):
                if not i == len(repeats) - 1:
                    path[repeats[i+1] : hit] = []
        return path

    def find_path(self):
        ants = self.init_ants()
        for i in range(self.iters):
            # print(i)
            for ant in ants:
                ant.setup()
                while not ant.final_node:
                    new_node = self.select_new_node(ant.curr, ant)
                    ant.move_ant(new_node)
                    ant.check_final(new_node)
                self.paths.append(self.remove_loops(ant.visited_nodes))
                ant.reset()
            self.update_weights()
            self.paths = []
            print('Iteration', i, "cost: ", len(self.best_path))
        return self.best_path

    def viz2(self):
        fig = go.Figure(go.Scatter(x=[self.start_coords[0]], y=[self.start_coords[1]], mode='markers', marker_color='red', name='Start'))
        fig.add_trace(go.Scatter(x=[self.end_coords[0]], y=[self.end_coords[1]], mode='markers', marker_color='green', name='End'))

        path_x, path_y = zip(*self.best_path)
        fig.add_trace(go.Scatter(x=path_y, y=path_x, mode='lines', line_color='red', name='Path'))

        fig.update_layout(
            title='Optimal Path',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
        )

        fig.add_trace(go.Heatmap(z=self.map, x0=0, y0=0, dx=1, dy=1, colorscale='temps'))

        return fig

    def get_best_path(self):
      return self.best_path

    def run(map_data, src, dest, iterations = 10, ants = 50, p = 0.5, early_stop = 1):
        aco = ACO(map_data, src, dest, iterations, ants, p, early_stop)
        best_path = aco.find_path()
        # print(best_path)
        fig = go.Figure(go.Scatter(x=[src[0]], y=[src[1]], mode='markers', marker_color='red', name='Start'))
        fig.add_trace(go.Scatter(x=[dest[0]], y=[dest[1]], mode='markers', marker_color='green', name='End'))

        path_x, path_y = zip(*best_path)
        fig.add_trace(go.Scatter(x=path_y, y=path_x, mode='lines', line_color='red', name='Path'))

        fig.update_layout(
            title='Optimal Path (ACO)',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
        )

        earth_elevation_colorscale = [
            [0.0, 'rgb(0, 0, 128)'],
            [0.1, 'rgb(0, 128, 255)'],
            [0.2, 'rgb(102, 204, 0)'],
            [0.4, 'rgb(0, 102, 0)'],
            [0.6, 'rgb(153, 153, 0)'],
            [0.8, 'rgb(153, 102, 51)'],
            [1.0, 'rgb(255, 255, 255)']
        ]


        fig.add_trace(go.Heatmap(z=map_data, x0=0, y0=0, dx=1, dy=1, colorscale=earth_elevation_colorscale))

        return best_path, fig

'''<------------------------------------------------------------------------------------------------------------------------------->'''

class Environment:
    def __init__(self, heatmap_data, state, goal):
        # self.grid = heatmap_data.values
        self.grid = heatmap_data
        self.num_rows, self.num_cols = self.grid.shape
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.state = state  # 0,0 start
        self.goal = goal  # end point finishing point
        self.max_reward = np.max(self.grid)
        # print(self.max_reward)
        self.reward_range = (0, self.max_reward)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        row, col = self.state
        reward = -1
        done = False

        new_row = row + action[0]
        new_col = col + action[1]
        if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols:
            self.state = (new_row, new_col)
        else:
            reward = -10  # Penalty

        if self.state == self.goal:
            reward = self.max_reward  # reward
            done = True

        return self.state, reward, done

class QLearningAgent:
    def __init__(
        self,
        environment,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
    ):  # discount factor is less nearer to 0 means it focuses on lower elevation from the start not caring about last steps reward
        self.environment = environment
        self.q_table = np.zeros(
            (
                environment.num_rows,
                environment.num_cols,
                len(environment.actions),
            )
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(len(self.environment.actions))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = (1 - self.learning_rate) * self.q_table[
            state[0], state[1], action
        ] + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state[0], state[1], action] = new_value

