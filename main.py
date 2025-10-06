import pygame
import random
import math
import neat
import pickle
import os
import tkinter as tk
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ecosystem Simulation with NEAT")

# Define colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_RED = (255, 182, 193)
SKY_BLUE = (135, 206, 235)

# Global variable to control vision range visualization
visualize_vision = False

# Global variables to store the last winning genomes for saving
last_winner_prey = None
last_winner_predator = None

# Initialize dictionaries to track the evolution of strategies over time
predator_strategy_counts = defaultdict(list)
prey_strategy_counts = defaultdict(list)


# Define GOAP components
class Action:
    def __init__(self, name, cost=1.0):
        self.name = name
        self.cost = cost
        self.preconditions = {}
        self.effects = {}

    def add_precondition(self, key, value):
        self.preconditions[key] = value

    def add_effect(self, key, value):
        self.effects[key] = value

    def is_achievable(self, state):
        for key, value in self.preconditions.items():
            if state.get(key) != value:
                return False
        return True


class Goal:
    def __init__(self, name, state):
        self.name = name
        self.state = state


class Node:
    def __init__(self, parent, action, state, cost, heuristic):
        self.parent = parent
        self.action = action
        self.state = state
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost


class Planner:
    def __init__(self, actions):
        self.actions = actions

    def plan(self, state, goal):
        open_list = []
        closed_list = set()
        start_node = Node(None, None, state, 0, self.heuristic(state, goal.state))
        open_list.append(start_node)

        while open_list:
            open_list.sort()
            current_node = open_list.pop(0)
            closed_list.add(tuple(current_node.state.items()))

            if self.goal_achieved(current_node.state, goal.state):
                return self.reconstruct_path(current_node)

            for action in self.actions:
                if action.is_achievable(current_node.state):
                    new_state = current_node.state.copy()
                    for key, value in action.effects.items():
                        new_state[key] = value
                    if tuple(new_state.items()) in closed_list:
                        continue
                    cost = current_node.cost + action.cost
                    heuristic = self.heuristic(new_state, goal.state)
                    new_node = Node(current_node, action, new_state, cost, heuristic)
                    open_list.append(new_node)

        # Fallback plan if no valid plan is found
        return [self.actions[-1]]  # Default action, assuming the last action is "Wander"

    def reconstruct_path(self, node):
        path = []
        while node.parent is not None:
            path.append(node.action)  # Ensure this is an Action object
            node = node.parent
        path.reverse()
        return path

    @staticmethod
    def goal_achieved(state, goal_state):
        for key, value in goal_state.items():
            if state.get(key) != value:
                return False
        return True

    @staticmethod
    def heuristic(state, goal_state):
        return sum(1 for key, value in goal_state.items() if state.get(key) != value)


class Shelter:
    def __init__(self):
        self.x = random.randint(100, width - 100)
        self.y = random.randint(100, height - 100)
        self.radius = 40

    def draw(self):
        points = []
        for i in range(8):
            angle = math.radians(45 * i)
            x = self.x + self.radius * math.cos(angle)
            y = self.y + self.radius * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.draw.polygon(screen, SKY_BLUE, points)


class PreyActions:
    def __init__(self):
        self.eat_action = Action("Eat", cost=1.0)
        self.eat_action.add_precondition("hungry", True)
        self.eat_action.add_effect("hungry", False)
        self.eat_action.add_effect("energy", 20)

        self.evade_action = Action("Evade", cost=2.0)
        self.evade_action.add_precondition("threat", True)
        self.evade_action.add_effect("safe", True)

        self.hide_action = Action("Hide", cost=0.5)
        self.hide_action.add_precondition("shelter_nearby", True)
        self.hide_action.add_precondition("threat", True)
        self.hide_action.add_effect("safe", True)

        self.wander_action = Action("Wander", cost=0.5)
        self.wander_action.add_precondition("indecisive", True)
        self.wander_action.add_effect("indecisive", False)

        self.actions = [self.eat_action, self.evade_action, self.hide_action, self.wander_action]

    def adjust_costs(self, success):
        if success:
            self.evade_action.cost *= 0.9  # Reduce cost by 10% for success
        else:
            self.evade_action.cost *= 1.1  # Increase cost by 10% for failure


class PredatorActions:
    def __init__(self):
        self.hunt_action = Action("Hunt", cost=1.0)
        self.hunt_action.add_precondition("hungry", True)
        self.hunt_action.add_effect("hungry", False)
        self.hunt_action.add_effect("energy", 50)

        self.coordinated_hunt_action = Action("Coordinated Hunt", cost=1.5)
        self.coordinated_hunt_action.add_precondition("hungry", True)
        self.coordinated_hunt_action.add_precondition("predator_nearby", True)
        self.coordinated_hunt_action.add_effect("hungry", False)
        self.coordinated_hunt_action.add_effect("energy", 75)

        self.wander_action = Action("Wander", cost=0.5)
        self.wander_action.add_precondition("indecisive", True)
        self.wander_action.add_effect("indecisive", False)

        self.actions = [self.hunt_action, self.coordinated_hunt_action, self.wander_action]

    def adjust_costs(self, success):
        if success:
            self.hunt_action.cost *= 0.9  # Reduce cost by 10% for success
            self.coordinated_hunt_action.cost *= 0.9
        else:
            self.hunt_action.cost *= 1.1  # Increase cost by 10% for failure
            self.coordinated_hunt_action.cost *= 1.1



class Plant:
    def __init__(self):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.energy = 50

    def draw(self):
        pygame.draw.circle(screen, GREEN, (self.x, self.y), 5)


class Predator:
    def __init__(self, genome, config):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.energy = 100
        self.min_energy = self.energy * 0.1  # Minimum energy level at 10%
        self.speed = 2.2
        self.detection_radius = 120
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.actions = PredatorActions()
        self.planner = Planner(self.actions.actions)
        self.hunt_success = 0  # Track successful hunts
        self.hunt_attempts = 0  # Track total hunt attempts
        self.wander_direction = (random.choice([-1, 1]), random.choice([-1, 1]))  # Set initial wander direction
        self.strategy_usage = defaultdict(int)  # To track strategy usage
        self.strategy_success = defaultdict(int)  # To track strategy success
        self.target_prey = None  # Target prey for coordinated hunt

    def update(self, prey_list, predator_list, shelters):
        if not prey_list:
            return

        # Find the closest entities
        nearby_prey = [prey for prey in prey_list if self.distance_to(prey) < self.detection_radius]
        nearby_predators = [pred for pred in predator_list if
                            pred != self and self.distance_to(pred) < self.detection_radius]

        closest_prey = min(nearby_prey, key=lambda p: self.distance_to(p), default=None)
        closest_predator = min(nearby_predators, key=lambda p: self.distance_to(p), default=None)
        closest_shelter = min(shelters, key=lambda s: self.distance_to(s), default=None)  # Closest shelter detection

        if closest_prey and closest_prey.in_shelter:
            # Skip attacking this prey, it is in a shelter
            closest_prey = None

        # Normalize the number of nearby predators and prey
        num_nearby_prey = len(nearby_prey) / 10  # Normalize by 10
        num_nearby_predators = len(nearby_predators) / 5  # Normalize by 5

        # Define the inputs for the neural network
        inputs = (
            self.x / width,
            self.y / height,
            self.energy / 100,
            (closest_prey.x / width if closest_prey else 0),
            (closest_prey.y / height if closest_prey else 0),
            (closest_predator.x / width if closest_predator else 0),  # Added input for closest predator detection
            (closest_predator.y / height if closest_predator else 0),  # Added input for closest predator detection
            (closest_shelter.x / width if closest_shelter else 0),  # New input for closest shelter detection
            (closest_shelter.y / height if closest_shelter else 0),  # New input for closest shelter detection
            num_nearby_prey,
            num_nearby_predators,  # Number of nearby predators
            0,
            0,
            0
        )

        output = self.net.activate(inputs)

        state = {
            "hungry": True,  # Always consider as "hungry" to keep hunting
            "prey_nearby": closest_prey is not None and self.distance_to(closest_prey) < self.detection_radius,
            "predator_nearby": closest_predator is not None,
        }

        # Adjust indecisiveness condition to be less sensitive
        state["indecisive"] = output[0] < 0.3 and not state["prey_nearby"]

        goal = Goal("Hunt", {"hungry": False})

        if state["indecisive"]:
            goal = Goal("Wander", {"indecisive": False})
        elif state["prey_nearby"] and state["predator_nearby"]:
            goal = Goal("Coordinated Hunt", {"hungry": False})

        plan = self.planner.plan(state, goal)

        if plan:
            action = plan[0]
            self.strategy_usage[action.name] += 1  # Track strategy usage

            if action.name == "Hunt" and closest_prey:
                self.hunt_attempts += 1  # Increment hunt attempts
                self.move_towards(closest_prey)
                if self.distance_to(closest_prey) < 10:
                    self.energy += closest_prey.energy
                    prey_list.remove(closest_prey)
                    self.hunt_success += 1  # Increment successful hunts
                    self.strategy_success[action.name] += 1  # Increment strategy success count

            elif action.name == "Coordinated Hunt" and closest_prey and closest_predator:
                self.perform_coordinated_hunt(closest_prey, closest_predator)

            elif action.name == "Wander":
                self.wander()

        # Adjust costs based on success
        self.actions.adjust_costs(success=self.hunt_success > 0)

        # Prevent energy from dropping below minimum level
        self.energy = max(self.min_energy, self.energy - 0.2)
        self.x = max(0, min(self.x, width))
        self.y = max(0, min(self.y, height))

    def perform_coordinated_hunt(self, prey, other_predator):
        # Determine the flanking position for this predator
        angle_to_prey = math.atan2(prey.y - self.y, prey.x - self.x)
        angle_offset = math.pi / 4 if random.random() < 0.5 else -math.pi / 4
        flanking_angle = angle_to_prey + angle_offset

        # Adjust the flanking position to avoid overlap
        target_x = prey.x + 50 * math.cos(flanking_angle)
        target_y = prey.y + 50 * math.sin(flanking_angle)
        self.move_towards((target_x, target_y))

        # Check if the predator is close enough to catch the prey
        if self.distance_to(prey) < 10:
            self.energy += prey.energy
            prey_list.remove(prey)
            self.hunt_success += 1  # Increment successful hunts
            self.strategy_success["Coordinated Hunt"] += 1  # Increment strategy success count
            print("Coordinated Hunt successful!")  # Debug statement
        else:
            # Even if not successful, still track the attempt
            self.strategy_usage["Coordinated Hunt"] += 1

    def wander(self):
        self.x += self.wander_direction[0] * self.speed
        self.y += self.wander_direction[1] * self.speed
        if random.random() < 0.1:  # Randomly change direction occasionally
            self.wander_direction = (random.choice([-1, 1]), random.choice([-1, 1]))

    def move_towards(self, target):
        if isinstance(target, tuple):
            target_x, target_y = target
        else:
            target_x, target_y = target.x, target.y

        angle = math.atan2(target_y - self.y, target_x - self.x)
        self.x += self.speed * math.cos(angle)
        self.y += self.speed * math.sin(angle)

    def move_away_from(self, target):
        if isinstance(target, tuple):
            target_x, target_y = target
        else:
            target_x, target_y = target.x, target.y

        angle = math.atan2(target_y - self.y, target_x - self.x)
        self.x -= self.speed * math.cos(angle)
        self.y -= self.speed * math.sin(angle)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 9)
        if visualize_vision:
            pygame.draw.circle(screen, LIGHT_RED, (int(self.x), int(self.y)), self.detection_radius, 1)




class Prey:
    def __init__(self, genome, config):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.energy = 60
        self.min_energy = self.energy * 0.1  # Minimum energy level at 10%
        self.speed = 2
        self.detection_radius = 85
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.actions = PreyActions()
        self.planner = Planner(self.actions.actions)
        self.evade_success = 0  # Track successful evasions
        self.evade_attempts = 0  # Track total evade attempts
        self.wander_direction = (random.choice([-1, 1]), random.choice([-1, 1]))  # Set initial wander direction
        self.strategy_usage = defaultdict(int)  # To track strategy usage
        self.strategy_success = defaultdict(int)  # To track strategy success
        self.in_shelter = False  # Track if the prey is hiding in a shelter

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def update(self, plants, predators, prey_list, shelters):
        # Get entities within detection range
        plants_in_range = [plant for plant in plants if self.distance_to(plant) < self.detection_radius]
        predators_in_range = [predator for predator in predators if self.distance_to(predator) < self.detection_radius]
        closest_plant = min(plants_in_range, key=lambda p: self.distance_to(p), default=None)
        closest_predator = min(predators_in_range, key=lambda p: self.distance_to(p), default=None)
        closest_shelter = min(shelters, key=lambda s: self.distance_to(s), default=None)  # Closest shelter detection
        closest_prey = min(
            [prey for prey in prey_list if prey != self and self.distance_to(prey) < self.detection_radius],
            key=lambda p: self.distance_to(p), default=None)
        num_prey_in_range = len(
            [p for p in prey_list if self.distance_to(p) < self.detection_radius and p != self]) / 10

        # Define the inputs for the neural network
        inputs = (
            self.x / width,
            self.y / height,
            self.energy / 100,
            (closest_plant.x / width if closest_plant else 0),
            (closest_plant.y / height if closest_plant else 0),
            (closest_predator.x / width if closest_predator else 0),
            (closest_predator.y / height if closest_predator else 0),
            (closest_prey.x / width if closest_prey else 0),
            (closest_prey.y / height if closest_prey else 0),
            num_prey_in_range,
            len(plants_in_range) / 10,  # Normalize the count of plants within range
            len(predators_in_range) / 5,  # Normalize the count of predators within range
            (closest_shelter.x / width if closest_shelter else 0),  # New input for closest shelter detection
            (closest_shelter.y / height if closest_shelter else 0)  # New input for closest shelter detection
        )

        output = self.net.activate(inputs)

        # Use the output to influence GOAP decision making
        state = {
            "hungry": True,  # Always consider as "hungry" to keep looking for food
            "threat": any(self.distance_to(p) < self.detection_radius for p in predators),
            "shelter_nearby": closest_shelter is not None and self.distance_to(closest_shelter) < self.detection_radius,
            "near_food": self.distance_to(closest_plant) < self.detection_radius if closest_plant else False
        }

        # Adjust indecisiveness condition to be less sensitive
        state["indecisive"] = output[0] < 0.3 and not state["threat"]

        goal = Goal("Survive", {"safe": True}) if state["threat"] else Goal("Eat", {"hungry": False})

        if state["indecisive"]:
            goal = Goal("Wander", {"indecisive": False})

        plan = self.planner.plan(state, goal)

        if plan:
            action = plan[0]  # Ensure this is an Action object
            self.strategy_usage[action.name] += 1  # Track strategy usage

            if action.name == "Eat" and closest_plant:
                self.move_towards(closest_plant)
                if self.distance_to(closest_plant) < 10:
                    self.energy += closest_plant.energy
                    plants.remove(closest_plant)
                    self.strategy_success[action.name] += 1  # Increment strategy success count
            elif action.name == "Evade" and closest_predator:
                self.evade_attempts += 1  # Increment evade attempts
                self.move_away_from(closest_predator)
                if self.distance_to(closest_predator) > self.detection_radius:
                    self.evade_success += 1  # Increment successful evasions
                    self.strategy_success[action.name] += 1  # Increment strategy success count
            elif action.name == "Hide" and closest_shelter:
                self.move_towards(closest_shelter)
                if self.distance_to(closest_shelter) < 10:
                    self.in_shelter = True  # Prey is now hiding in shelter
                    self.strategy_success[action.name] += 1  # Increment strategy success count
            elif action.name == "Wander":
                if not self.in_shelter:  # Only wander if not hiding
                    self.wander()

        # Adjust costs based on success
        self.actions.adjust_costs(success=self.evade_success > 0)

        # Prevent energy from dropping below minimum level
        self.energy = max(self.min_energy, self.energy - 0.1)
        self.x = max(0, min(self.x, width))
        self.y = max(0, min(self.y, height))

        # If no predators are nearby, leave the shelter
        if self.in_shelter and not state["threat"]:
            self.in_shelter = False

    def wander(self):
        self.x += self.wander_direction[0] * self.speed
        self.y += self.wander_direction[1] * self.speed
        if random.random() < 0.1:  # Randomly change direction occasionally
            self.wander_direction = (random.choice([-1, 1]), random.choice([-1, 1]))

    def move_towards(self, target):
        angle = math.atan2(target.y - self.y, target.x - self.x)
        self.x += self.speed * math.cos(angle)
        self.y += self.speed * math.sin(angle)

    def move_away_from(self, target):
        if isinstance(target, tuple):
            target_x, target_y = target
        else:
            target_x, target_y = target.x, target.y

        angle = math.atan2(target_y - self.y, target_x - self.x)
        self.x -= self.speed * math.cos(angle)
        self.y -= self.speed * math.sin(angle)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def draw(self):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 7)
        if visualize_vision:
            pygame.draw.circle(screen, LIGHT_BLUE, (int(self.x), int(self.y)), self.detection_radius, 1)


# Ensure that the eval_genomes function is correctly set up
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        plants = [Plant() for _ in range(50)]
        shelters = [Shelter() for _ in range(5)]
        prey = Prey(genome, config)
        predators = [Predator(genome, config) for _ in range(10)]

        fitness = 0
        steps = 500  # Number of simulation steps
        for step in range(steps):
            if len(plants) < 20:
                plants.append(Plant())

            # Update prey and calculate fitness based on GOAP plan success
            prey_state = {
                "hungry": prey.energy < 40,
                "threat": any(prey.distance_to(p) < prey.detection_radius for p in predators),
                "near_food": any(prey.distance_to(p) < prey.detection_radius for p in plants),
                "shelter_nearby": any(prey.distance_to(s) < prey.detection_radius for s in shelters)
            }
            prey_goal = Goal("Survive", {"safe": True}) if prey.energy < 40 else Goal("Eat", {"hungry": False})
            prey_plan = prey.planner.plan(prey_state, prey_goal)
            action_taken = False
            if prey_plan:
                action = prey_plan[0]
                action_taken = True
                if action.name == "Eat":
                    closest_plant = min(plants, key=lambda p: prey.distance_to(p), default=None)
                    if closest_plant:
                        prey.move_towards(closest_plant)
                        if prey.distance_to(closest_plant) < 10:
                            prey.energy += closest_plant.energy
                            plants.remove(closest_plant)
                            fitness += 50  # Reward for successful eating
                elif action.name == "Evade":
                    closest_predator = min(predators, key=lambda p: prey.distance_to(p), default=None)
                    if closest_predator:
                        prey.move_away_from(closest_predator)
                        fitness += 30  # Reward for successful evasion

            # Penalize for not taking any action (indecisiveness)
            if not action_taken:
                fitness -= 5

            # General reward for surviving
            fitness += prey.energy  # Use original fitness approach

            # Update predators and calculate fitness based on GOAP plan success
            for predator in predators[:]:
                predator_state = {
                    "hungry": predator.energy < 50,
                    "prey_nearby": any(predator.distance_to(p) < predator.detection_radius for p in [prey])
                }
                predator_goal = Goal("Hunt", {"hungry": False})
                predator_plan = predator.planner.plan(predator_state, predator_goal)
                if predator_plan:
                    action = predator_plan[0]
                    if action.name == "Hunt":
                        predator.move_towards(prey)
                        if predator.distance_to(prey) < 10:
                            predator.energy += prey.energy
                            fitness += 100  # Reward for successful hunting
                            prey.energy = 0  # Prey is caught, no energy left
                            break  # Prey is caught, exit loop

            if prey.energy <= prey.min_energy:
                break

        genome.fitness = fitness


def run_neat(config_path):
    global last_winner_prey
    global last_winner_predator

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, 50)

    last_winner_prey = winner  # Save the winner for prey
    last_winner_predator = winner  # Save the winner for predator

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

    print(f'\nBest genome:\n{winner}')


# Save functions for prey and predator
def save_prey_data():
    global last_winner_prey
    if last_winner_prey:
        with open('winner_prey.pkl', 'wb') as f:
            pickle.dump(last_winner_prey, f)
        print("Prey training data saved.")
    else:
        print("No prey data to save.")


def save_predator_data():
    global last_winner_predator
    if last_winner_predator:
        with open('winner_predator.pkl', 'wb') as f:
            pickle.dump(last_winner_predator, f)
        print("Predator training data saved.")
    else:
        print("No predator data to save.")


# Hunting Grounds Arrays
def create_random_hunting_ground(num_prey, num_predators):
    prey_positions = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_prey)]
    predator_positions = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_predators)]
    return prey_positions, predator_positions


def create_clustered_hunting_ground(num_prey, num_predators):
    prey_center = (width // 4, height // 4)
    predator_center = (3 * width // 4, height // 4)

    prey_positions = [(prey_center[0] + random.randint(-50, 50), prey_center[1] + random.randint(-50, 50)) for _ in
                      range(num_prey)]
    predator_positions = [(predator_center[0] + random.randint(-50, 50), predator_center[1] + random.randint(-50, 50))
                          for _ in range(num_predators)]
    return prey_positions, predator_positions


def create_symmetric_hunting_ground(num_prey, num_predators):
    prey_positions = [(i * (width // num_prey), height // 4) for i in range(num_prey)]
    predator_positions = [(i * (width // num_predators), 3 * height // 4) for i in range(num_predators)]
    return prey_positions, predator_positions


# Predefined shelter layouts
def create_predefined_shelter_layout1():
    shelters = [
        Shelter(), Shelter(), Shelter(), Shelter(), Shelter()
    ]
    shelters[0].x, shelters[0].y = width // 4, height // 4
    shelters[1].x, shelters[1].y = 3 * width // 4, height // 4
    shelters[2].x, shelters[2].y = width // 4, 3 * height // 4
    shelters[3].x, shelters[3].y = 3 * width // 4, 3 * height // 4
    shelters[4].x, shelters[4].y = width // 2, height // 2
    return shelters


def create_predefined_shelter_layout2():
    shelters = [
        Shelter(), Shelter(), Shelter()
    ]
    shelters[0].x, shelters[0].y = width // 2, height // 4
    shelters[1].x, shelters[1].y = width // 4, height // 2
    shelters[2].x, shelters[2].y = 3 * width // 4, height // 2
    return shelters


def create_predefined_shelter_layout0():
    shelters = []
    return shelters


# Multi-line Plot for Strategies
def plot_strategy_usage(strategy_usage_data, entity_type="Prey"):
    if not strategy_usage_data:
        print(f"No strategy data to analyze for {entity_type}.")
        return

    # Create a dictionary to store the usage of each strategy over time
    strategy_over_time = defaultdict(list)

    for time_step, usage in enumerate(strategy_usage_data):
        for strategy, count in usage.items():
            strategy_over_time[strategy].append((time_step, count))

    # Plot the strategies over time
    plt.figure(figsize=(8, 4))
    for strategy, usage_data in strategy_over_time.items():
        time_steps, counts = zip(*usage_data)
        plt.plot(time_steps, counts, label=strategy)

    plt.title(f'Strategy Usage Over Time for {entity_type}')
    plt.xlabel('Time Steps')
    plt.ylabel('Usage Count')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot for most successful strategy
def plot_most_successful_strategy(strategy_success_data, entity_type="Prey"):
    if not strategy_success_data:
        print(f"No strategy success data to analyze for {entity_type}.")
        return

    strategy_success_count = defaultdict(int)
    for success in strategy_success_data:
        for strategy, count in success.items():
            strategy_success_count[strategy] += count

    # Sort strategies by success count
    strategies = sorted(strategy_success_count.keys(), key=lambda x: strategy_success_count[x], reverse=True)

    plt.figure(figsize=(8, 4))
    plt.bar(strategies, [strategy_success_count[strategy] for strategy in strategies], color='green')
    plt.title(f'Most Successful Strategies for {entity_type}')
    plt.xlabel('Strategy')
    plt.ylabel('Success Count')
    plt.grid(True)
    plt.show()


# Plot for most reliable strategy
def plot_most_reliable_strategy(strategy_usage_data, strategy_success_data, entity_type="Prey"):
    if not strategy_usage_data or not strategy_success_data:
        print(f"No strategy reliability data to analyze for {entity_type}.")
        return

    strategy_reliability = defaultdict(float)
    strategy_total_usage = defaultdict(int)
    strategy_total_success = defaultdict(int)

    for usage, success in zip(strategy_usage_data, strategy_success_data):
        for strategy in usage.keys():
            strategy_total_usage[strategy] += usage.get(strategy, 0)
            strategy_total_success[strategy] += success.get(strategy, 0)

    for strategy in strategy_total_usage.keys():
        if strategy_total_usage[strategy] > 0:
            strategy_reliability[strategy] = strategy_total_success[strategy] / strategy_total_usage[strategy]

    # Sort strategies by reliability
    strategies = sorted(strategy_reliability.keys(), key=lambda x: strategy_reliability[x], reverse=True)

    plt.figure(figsize=(8, 4))
    plt.bar(strategies, [strategy_reliability[strategy] for strategy in strategies], color='blue')
    plt.title(f'Most Reliable Strategies for {entity_type}')
    plt.xlabel('Strategy')
    plt.ylabel('Reliability (Success/Usage)')
    plt.grid(True)
    plt.show()


def run_simulation(winner, config, num_prey, num_predators, prey_positions, predator_positions, shelter_layout):
    plants = [Plant() for _ in range(50)]
    prey_list = [Prey(winner, config) for _ in range(num_prey)]
    predators = [Predator(winner, config) for _ in range(num_predators)]

    if shelter_layout == 1:
        shelters = create_predefined_shelter_layout1()
    elif shelter_layout == 2:
        shelters = create_predefined_shelter_layout2()
    elif shelter_layout == 0:
        shelters = create_predefined_shelter_layout0()
    else:
        shelters = [Shelter() for _ in range(5)]

    # Assign positions from the hunting ground layout
    for prey, pos in zip(prey_list, prey_positions):
        prey.x, prey.y = pos

    for predator, pos in zip(predators, predator_positions):
        predator.x, predator.y = pos

    clock = pygame.time.Clock()
    running = True

    # Initialize font for displaying text
    font = pygame.font.SysFont(None, 24)

    # Initialize lists to collect efficiency data over time
    predator_hunting_efficiency = []
    prey_evading_efficiency = []
    time_steps = []

    # Track strategy usage over time
    prey_strategy_usage_over_time = []
    predator_strategy_usage_over_time = []

    # Track strategy success over time
    prey_strategy_success_over_time = []
    predator_strategy_success_over_time = []

    step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # End the simulation if all prey are dead
        if not prey_list:
            print("All prey are dead. Ending simulation.")
            break

        screen.fill(WHITE)

        # Increase the plant regeneration rate and ensure a minimum number of plants
        if len(plants) < 20:
            plants.extend([Plant() for _ in range(20)])  # Add more plants to replenish faster

        for plant in plants[:]:
            plant.draw()

        for shelter in shelters:
            shelter.draw()

        prey_strategy_usage = defaultdict(int)
        prey_strategy_success = defaultdict(int)
        for prey in prey_list[:]:
            prey.update(plants, predators, prey_list, shelters)
            prey.draw()
            # Track strategy usage and success
            for strategy, count in prey.strategy_usage.items():
                prey_strategy_usage[strategy] += count
            for strategy, count in prey.strategy_success.items():
                prey_strategy_success[strategy] += count

        predator_strategy_usage = defaultdict(int)
        predator_strategy_success = defaultdict(int)
        for predator in predators[:]:
            predator.update(prey_list, predators, shelters)  # Pass both prey_list and predator_list
            predator.draw()
            # Track strategy usage and success
            for strategy, count in predator.strategy_usage.items():
                predator_strategy_usage[strategy] += count
            for strategy, count in predator.strategy_success.items():
                predator_strategy_success[strategy] += count

        # Store the strategy usage and success for this step
        prey_strategy_usage_over_time.append(prey_strategy_usage)
        prey_strategy_success_over_time.append(prey_strategy_success)
        predator_strategy_usage_over_time.append(predator_strategy_usage)
        predator_strategy_success_over_time.append(predator_strategy_success)

        # Display the prey and predator counts
        prey_count_text = font.render(f'Prey: {len(prey_list)}', True, BLACK)
        predator_count_text = font.render(f'Predators: {len(predators)}', True, BLACK)

        screen.blit(prey_count_text, (10, 10))  # Display at top-left
        screen.blit(predator_count_text, (10, 50))  # Display below prey count

        pygame.display.flip()
        clock.tick(30)

        # Collect efficiency data
        predator_hunt_efficiency = sum(
            predator.hunt_success / max(1, predator.hunt_attempts) for predator in predators) / len(predators)
        prey_evade_efficiency = sum(
            prey.evade_success / prey.evade_attempts if prey.evade_attempts > 0 else 0 for prey in prey_list) / len(
            prey_list) if prey_list else 0

        predator_hunting_efficiency.append(predator_hunt_efficiency)
        prey_evading_efficiency.append(prey_evade_efficiency)
        time_steps.append(step_count)

        step_count += 1

    pygame.quit()

    # Plot the efficiency graphs
    plot_efficiencies(time_steps, predator_hunting_efficiency, prey_evading_efficiency)

    # Plot strategy usage over time for prey and predators
    print("Plotting strategy usage for prey.")
    plot_strategy_usage(prey_strategy_usage_over_time, entity_type="Prey")
    print("Plotting strategy usage for predators.")
    plot_strategy_usage(predator_strategy_usage_over_time, entity_type="Predator")

    # Plot most successful strategies for prey and predators
    print("Plotting most successful strategies for prey.")
    plot_most_successful_strategy(prey_strategy_success_over_time, entity_type="Prey")
    print("Plotting most successful strategies for predators.")
    plot_most_successful_strategy(predator_strategy_success_over_time, entity_type="Predator")

    # Plot most reliable strategies for prey and predators
    print("Plotting most reliable strategies for prey.")
    plot_most_reliable_strategy(prey_strategy_usage_over_time, prey_strategy_success_over_time, entity_type="Prey")
    print("Plotting most reliable strategies for predators.")
    plot_most_reliable_strategy(predator_strategy_usage_over_time, predator_strategy_success_over_time, entity_type="Predator")



def plot_efficiencies(time_steps, predator_hunting_efficiency, prey_evading_efficiency):
    plt.figure(figsize=(10, 6))

    # Plot Predator Hunting Efficiency as Bar Graph
    plt.subplot(2, 1, 1)
    plt.bar(time_steps, predator_hunting_efficiency, color='red', label='Hunting Efficiency')
    plt.xlabel('Time Steps')
    plt.ylabel('Hunting Efficiency')
    plt.title('Predator Hunting Efficiency Over Time')
    plt.legend()
    plt.grid(True)

    # Plot Prey Evading Efficiency as Bar Graph
    plt.subplot(2, 1, 2)
    plt.bar(time_steps, prey_evading_efficiency, color='blue', label='Evading Efficiency')
    plt.xlabel('Time Steps')
    plt.ylabel('Evading Efficiency')
    plt.title('Prey Evading Efficiency Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Tkinter GUI Setup
def start_simulation():
    try:
        num_prey = int(prey_entry.get())
        num_predators = int(predators_entry.get())
        global visualize_vision
        visualize_vision = vision_var.get()

        if num_prey <= 0 or num_predators <= 0:
            print("Invalid Input: Please enter positive numbers for prey and predators.")
            return

        hunting_ground_type = hunting_ground_var.get()
        shelter_layout = int(shelter_layout_var.get())

        # Ensure any running Pygame instance is properly quit before starting a new one
        pygame.display.quit()  # Close the Pygame window
        pygame.quit()  # Quit Pygame

        # Reinitialize Pygame and the display
        pygame.init()
        global screen
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ecosystem Simulation with NEAT")

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')

        run_neat(config_path)

        with open('winner.pkl', 'rb') as f:
            winner = pickle.load(f)

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        # Determine hunting ground layout
        if hunting_ground_type == 'clustered':
            prey_positions, predator_positions = create_clustered_hunting_ground(num_prey, num_predators)
        elif hunting_ground_type == 'symmetric':
            prey_positions, predator_positions = create_symmetric_hunting_ground(num_prey, num_predators)
        else:  # 'random'
            prey_positions, predator_positions = create_random_hunting_ground(num_prey, num_predators)

        run_simulation(winner, config, num_prey, num_predators, prey_positions, predator_positions, shelter_layout)
    except ValueError:
        print("Invalid Input: Please enter valid numbers for prey and predators.")


# Create Tkinter window
root = tk.Tk()
root.title("Simulation Setup")

# Number of Prey input
tk.Label(root, text="Number of Prey:").grid(row=0, column=0, padx=10, pady=10)
prey_entry = tk.Entry(root)
prey_entry.grid(row=0, column=1, padx=10, pady=10)
prey_entry.insert(0, "20")  # Set default value for prey

# Number of predators input
tk.Label(root, text="Number of Predators:").grid(row=1, column=0, padx=10, pady=10)
predators_entry = tk.Entry(root)
predators_entry.grid(row=1, column=1, padx=10, pady=10)
predators_entry.insert(0, "10")  # Set default value for predators

# Hunting ground type input using radio buttons
tk.Label(root, text="Hunting Ground Type:").grid(row=2, column=0, padx=10, pady=10)
hunting_ground_var = tk.StringVar(value='random')

# Random hunting ground
random_rb = tk.Radiobutton(root, text="Random", variable=hunting_ground_var, value='random')
random_rb.grid(row=2, column=1, padx=10, pady=5)

# Clustered hunting ground
clustered_rb = tk.Radiobutton(root, text="Clustered", variable=hunting_ground_var, value='clustered')
clustered_rb.grid(row=3, column=1, padx=10, pady=5)

# Symmetric hunting ground
symmetric_rb = tk.Radiobutton(root, text="Symmetric", variable=hunting_ground_var, value='symmetric')
symmetric_rb.grid(row=4, column=1, padx=10, pady=5)

# Shelter layout option using radio buttons
tk.Label(root, text="Shelter Layout:").grid(row=5, column=0, padx=10, pady=10)
shelter_layout_var = tk.StringVar(value='1')

layout1_rb = tk.Radiobutton(root, text="Layout 1", variable=shelter_layout_var, value='1')
layout1_rb.grid(row=5, column=1, padx=10, pady=5)

layout2_rb = tk.Radiobutton(root, text="Layout 2", variable=shelter_layout_var, value='2')
layout2_rb.grid(row=6, column=1, padx=10, pady=5)

layout0_rb = tk.Radiobutton(root, text="No Shelter", variable=shelter_layout_var, value='0')
layout0_rb.grid(row=7, column=1, padx=10, pady=5)

# Vision range visualization checkbox
vision_var = tk.BooleanVar()
tk.Checkbutton(root, text="Visualize Vision Range", variable=vision_var).grid(row=8, columnspan=2)

# Buttons to save prey and predator data
save_prey_button = tk.Button(root, text="Save Prey Data", command=save_prey_data)
save_prey_button.grid(row=9, column=0, pady=20)

save_predator_button = tk.Button(root, text="Save Predator Data", command=save_predator_data)
save_predator_button.grid(row=9, column=1, pady=20)

# Start button
start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.grid(row=10, columnspan=2, pady=20)

root.mainloop()
