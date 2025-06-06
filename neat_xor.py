import numpy as np
import random
import os
import logging
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


# XOR dataset
# The XOR (exclusive OR) operation is a logical operation that outputs 1 (true)
# only when the two input values are different.
# If both inputs are the same (both 0 or both 1), the output is 0 (false).
data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0)
]

# Activation function
sigmoid = lambda x: 1 / (1 + np.exp(-x))

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class NodeGene:
    def __init__(self, id, type):
        self.id = id
        self.type = type  # 'input', 'hidden', 'output'
        self.value = 0.0

class Genome:
    def __init__(self, num_inputs, num_outputs):
        self.nodes = []
        self.connections = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fitness = None
        self.next_node_id = 0
        # Create input and output nodes
        for _ in range(num_inputs):
            self.nodes.append(NodeGene(self.next_node_id, 'input'))
            self.next_node_id += 1
        # Add two hidden nodes at the start
        hidden_ids = []
        for _ in range(2):
            hidden_node = NodeGene(self.next_node_id, 'hidden')
            self.nodes.append(hidden_node)
            hidden_ids.append(self.next_node_id)
            self.next_node_id += 1
        output_ids = []
        for _ in range(num_outputs):
            self.nodes.append(NodeGene(self.next_node_id, 'output'))
            output_ids.append(self.next_node_id)
            self.next_node_id += 1
        # Fully connect input to hidden, hidden to output
        innovation = 0
        for i in range(num_inputs):
            for h in hidden_ids:
                self.connections.append(ConnectionGene(i, h, np.random.uniform(-0.7, 0.7), True, innovation))
                innovation += 1
        for h in hidden_ids:
            for output_id in output_ids:
                self.connections.append(ConnectionGene(h, output_id, np.random.uniform(-0.7, 0.7), True, innovation))
                innovation += 1
        # Optionally, also connect input to output directly (classic NEAT)
        for i in range(num_inputs):
            for output_id in output_ids:
                self.connections.append(ConnectionGene(i, output_id, np.random.uniform(-0.7, 0.7), True, innovation))
                innovation += 1

    def feed_forward(self, x):
        node_dict = {node.id: node for node in self.nodes}
        # Set input node values
        for i in range(self.num_inputs):
            node_dict[i].value = x[i]
        # Topological sort: compute order of nodes
        # Start with input nodes, then hidden, then output
        order = [n.id for n in self.nodes if n.type == 'input']
        order += [n.id for n in self.nodes if n.type == 'hidden']
        order += [n.id for n in self.nodes if n.type == 'output']
        # Compute node values in topological order
        for nid in order:
            node = node_dict[nid]
            if node.type == 'input':
                continue
            s = 0.0
            for conn in self.connections:
                if conn.enabled and conn.out_node == nid:
                    if conn.in_node in node_dict:
                        s += node_dict[conn.in_node].value * conn.weight
            node.value = sigmoid(s)
        return [node_dict[nid].value for nid in order if node_dict[nid].type == 'output']

    def mutate(self):
        # Mutate weights (smaller step)
        for conn in self.connections:
            if random.random() < 0.8:
                conn.weight += np.random.normal(0, 0.2)
        # Add connection (higher probability)
        if random.random() < 0.2:
            possible_in = [n.id for n in self.nodes]
            possible_out = [n.id for n in self.nodes if n.type != 'input']
            in_node = random.choice(possible_in)
            out_node = random.choice(possible_out)
            # Prevent duplicate connections and self-loops
            if in_node != out_node and not any((c.in_node == in_node and c.out_node == out_node) for c in self.connections):
                # Prevent cycles: only add if there is no path from out_node to in_node using enabled connections
                from collections import deque
                node_graph = {n.id: [] for n in self.nodes}
                for c in self.connections:
                    if c.enabled and c.in_node in node_graph and c.out_node in node_graph:
                        node_graph[c.in_node].append(c.out_node)
                # BFS to check if in_node is reachable from out_node (using only enabled connections)
                queue = deque([out_node])
                visited = set()
                cycle = False
                while queue:
                    current = queue.popleft()
                    if current == in_node:
                        cycle = True
                        break
                    for neighbor in node_graph.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if not cycle:
                    max_innov = max([c.innovation for c in self.connections], default=-1)
                    self.connections.append(ConnectionGene(in_node, out_node, np.random.uniform(-0.7, 0.7), True, max_innov+1))
        # Add node (higher probability)
        if random.random() < 0.2 and self.connections:
            conn = random.choice([c for c in self.connections if c.enabled])
            if conn:
                conn.enabled = False
                new_node = NodeGene(self.next_node_id, 'hidden')
                self.nodes.append(new_node)
                max_innov = max([c.innovation for c in self.connections], default=-1)
                self.connections.append(ConnectionGene(conn.in_node, new_node.id, 1.0, True, max_innov+1))
                self.connections.append(ConnectionGene(new_node.id, conn.out_node, conn.weight, True, max_innov+2))
                self.next_node_id += 1

    def crossover(self, other):
        child = Genome(self.num_inputs, self.num_outputs)
        child.nodes = [NodeGene(n.id, n.type) for n in self.nodes]
        child.connections = []
        for c in self.connections:
            # Prevent self-loops in crossover
            if c.in_node == c.out_node:
                continue
            match = next((oc for oc in other.connections if oc.innovation == c.innovation), None)
            if match and random.random() < 0.5:
                if match.in_node != match.out_node:
                    child.connections.append(ConnectionGene(match.in_node, match.out_node, match.weight, match.enabled, match.innovation))
            else:
                child.connections.append(ConnectionGene(c.in_node, c.out_node, c.weight, c.enabled, c.innovation))
        child.next_node_id = self.next_node_id
        # Remove cycles if any
        child._remove_cycles()
        return child

    def _remove_cycles(self):
        # Remove cycles by disabling one connection in each cycle until acyclic
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node(n.id)
        for c in self.connections:
            if c.enabled:
                G.add_edge(c.in_node, c.out_node)
        try:
            cycles = list(nx.simple_cycles(G))
            while cycles:
                cycle = cycles[0]
                if len(cycle) < 2:
                    logging.error(f"Malformed or empty cycle detected: {cycle}. Skipping.")
                    break
                # Handle self-loop
                if len(cycle) == 1:
                    u = v = cycle[0]
                else:
                    u, v = cycle[0], cycle[1]
                found = False
                for c in self.connections:
                    if c.enabled and c.in_node == u and c.out_node == v:
                        c.enabled = False
                        found = True
                        break
                if not found:
                    logging.error(f"Could not find edge to disable for cycle: {cycle}")
                    break
                # Rebuild the graph
                G = nx.DiGraph()
                for n in self.nodes:
                    G.add_node(n.id)
                for c in self.connections:
                    if c.enabled:
                        G.add_edge(c.in_node, c.out_node)
                cycles = list(nx.simple_cycles(G))
        except Exception as e:
            logging.error(f"Error during cycle removal: {e}")

class Species:
    def __init__(self, representative):
        self.representative = representative  # Genome
        self.members = [representative]
        self.best_fitness = representative.fitness
        self.stagnant = 0

    def add(self, genome):
        self.members.append(genome)

    def reset(self):
        self.members = []

# Genetic distance function (very simple)
def genome_distance(g1, g2, c1=1.0, c2=0.4):
    # Count disjoint/excess genes and average weight difference for matching genes
    innov1 = {c.innovation: c for c in g1.connections}
    innov2 = {c.innovation: c for c in g2.connections}
    all_innov = set(innov1.keys()).union(innov2.keys())
    disjoint = 0
    weight_diff = 0
    matches = 0
    for i in all_innov:
        c1g = innov1.get(i)
        c2g = innov2.get(i)
        if c1g and c2g:
            weight_diff += abs(c1g.weight - c2g.weight)
            matches += 1
        else:
            disjoint += 1
    avg_weight_diff = weight_diff / matches if matches > 0 else 0
    return c1 * disjoint + c2 * avg_weight_diff

def speciate(population, species_list, threshold=2.0):
    for s in species_list:
        s.reset()
    for genome in population:
        found = False
        for s in species_list:
            if genome_distance(genome, s.representative) < threshold:
                s.add(genome)
                found = True
                break
        if not found:
            species_list.append(Species(genome))
    # Remove empty species
    species_list[:] = [s for s in species_list if s.members]

def evaluate_fitness(genome):
    error = 0.0
    for x, y in data:
        output = genome.feed_forward(x)[0]
        error += (output - y) ** 2
    # Regularization penalties
    node_penalty = 0.001  # Penalty per node
    conn_penalty = 0.001  # Penalty per enabled connection
    num_nodes = len(genome.nodes)
    num_enabled_conns = len([c for c in genome.connections if c.enabled])
    genome.fitness = 4 - error - node_penalty * num_nodes - conn_penalty * num_enabled_conns
    return genome.fitness

def run_neat(generations=100, pop_size=200):
    import json
    os.makedirs('output', exist_ok=True)
    population = [Genome(2, 1) for _ in range(pop_size)]
    species_list = []
    progress = []
    for gen in range(generations):
        for genome in population:
            evaluate_fitness(genome)
        speciate(population, species_list, threshold=2.0)
        for s in species_list:
            s.members.sort(key=lambda g: g.fitness if g.fitness is not None else -float('inf'), reverse=True)
            if s.members[0].fitness > (s.best_fitness if hasattr(s, 'best_fitness') else -float('inf')):
                s.best_fitness = s.members[0].fitness
                s.stagnant = 0
            else:
                s.stagnant += 1
        species_list[:] = [s for s in species_list if s.stagnant < 30]
        if not species_list:
            logging.info(f"Gen {gen}: All species extinct. Stopping early.")
            break
        best_genome = max(population, key=lambda g: g.fitness if g.fitness is not None else -float('inf'))
        best_fitness = best_genome.fitness
        logging.info(f"Gen {gen}: Species {len(species_list)}, Best fitness {best_fitness:.3f}")
        # Save all info needed for plotting
        progress.append({
            'generation': gen,
            'best_fitness': best_fitness,
            'nodes': [
                {'id': n.id, 'type': n.type} for n in best_genome.nodes
            ],
            'connections': [
                {'in_node': c.in_node, 'out_node': c.out_node, 'weight': c.weight, 'enabled': c.enabled, 'innovation': c.innovation}
                for c in best_genome.connections
            ]
        })
        # Elitism: keep top 3 from each species
        new_pop = []
        for s in species_list:
            new_pop.extend(s.members[:3])
        while len(new_pop) < pop_size:
            s = random.choice(species_list)
            parent1 = random.choice(s.members[:max(1, len(s.members)//2)])
            parent2 = random.choice(s.members[:max(1, len(s.members)//2)])
            child = parent1.crossover(parent2)
            child.mutate()
            new_pop.append(child)
        population = new_pop[:pop_size]
    # Save progress as JSON
    with open('output/progress.json', 'w') as f:
        json.dump(progress, f, indent=2)
    # Test best genome
    best = max(population, key=lambda g: g.fitness if g.fitness is not None else -float('inf'))
    logging.info("\nBest Genome Network Structure:")
    logging.info("Nodes:")
    for node in best.nodes:
        logging.info(f"  id={node.id}, type={node.type}")
    logging.info("Connections:")
    for conn in best.connections:
        if conn.enabled:
            logging.info(f"  {conn.in_node} -> {conn.out_node} (weight={conn.weight:.3f}, innovation={conn.innovation})")
    for x, y in data:
        out = best.feed_forward(x)[0]
        logging.info(f"Input: {x}, Predicted: {out:.3f}, Target: {y}")

if __name__ == "__main__":
    run_neat()
