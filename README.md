# NEAT-GOAP: Evolving Intelligent Agents with Goal-Oriented Planning

This research project explores the integration of NeuroEvolution of Augmented Topologies (NEAT) with Goal-Oriented Action Planning (GOAP) to create intelligent agents capable of developing evolutionarily stable strategies in dynamic predator-prey simulations. The system evolves neural networks that process environmental information and make real-time decisions through goal-oriented planning, enabling the emergence of complex adaptive behaviors.

## Project Overview

The core innovation of this work lies in combining two powerful AI techniques: NEAT, which evolves neural network structures through genetic algorithms, and GOAP, which enables dynamic action planning based on changing goals and environmental conditions. This hybrid approach allows agents to not only react to their environment but also develop sophisticated, long-term strategies that become stable within the population over time.

In the simulation ecosystem, prey agents must balance finding food while avoiding predators, while predators must develop effective hunting strategies. The system tests how different environmental factors—specifically agent spawning patterns and shelter availability—influence which strategies become dominant and persist in the population.

## Key Components

### Neural Network Evolution with NEAT
The system uses NEAT to evolve neural networks that process sensory inputs from the environment. Each agent has specific input nodes tailored to its role: prey detect nearby plants, predators, shelters, and other prey, while predators track prey locations, shelter positions, and other predators. These neural networks grow in complexity over generations, allowing agents to develop increasingly sophisticated decision-making capabilities.

### Goal-Oriented Action Planning
GOAP provides the planning framework where agents dynamically select goals based on their current situation and generate action sequences to achieve those goals. The system includes a novel dynamic cost adjustment feature where successful plans have their costs reduced over time, reinforcing effective strategies.

### Simulation Environment
The project implements a predator-prey ecosystem using Pygame for visualization. The environment includes:
- Prey agents that must eat plants to survive while avoiding predators
- Predator agents that hunt prey for sustenance  
- Shelter structures that provide protection for prey
- Multiple spawning patterns (symmetric, cluster, random) and shelter layouts (none, minimalist, spread)

## Research Findings

Through nine distinct experimental scenarios combining different spawning patterns and shelter layouts, the system demonstrated how evolutionarily stable strategies emerge and persist. The presence of shelters significantly influenced strategy stability, with hiding becoming the dominant strategy for prey when adequate shelter was available, while evasion served as the secondary strategy.

Predators adapted by learning to wait near shelters or hunt isolated prey. The NEAT-GOAP system enabled dynamic strategy adjustment, with agents learning from environmental feedback and refining their decision-making processes over time. Notably, the system produced emergent behaviors like group hunting that weren't explicitly programmed.

## Technical Implementation

The architecture processes environmental information through evolved neural networks, which inform the GOAP planner about the agent's current situation. The planner then selects appropriate goals and generates action plans using an A* search algorithm. Agents execute these plans while continuously monitoring for changing conditions that might require replanning.

The system includes comprehensive metrics for evaluation: hunting and evading efficiency, strategy usage over time, success rates of different strategies, and reliability measurements (success-to-usage ratios).

## Getting Started

To run the simulations, you'll need Python with Pygame installed. The project includes configurations for nine different experimental scenarios that combine various spawning patterns and shelter layouts. Each scenario can be configured through the simulation interface to observe how different environmental conditions affect strategy evolution.

## Future Directions

This research opens several avenues for future work, including integration with Monte Carlo Tree Search for better decision space exploration, experimentation with spiking neural networks for temporal information processing, and applications in robotics and autonomous systems where real-time decision making and strategy evolution are critical.

The NEAT-GOAP framework demonstrates significant potential for developing adaptive AI systems that can operate effectively in dynamic, unpredictable environments while developing stable, effective strategies through evolutionary processes.

## Reference

This project was developed as part of dissertation research at the University of Sussex, exploring the intersection of evolutionary algorithms, planning systems, and game theory in multi-agent simulations.
