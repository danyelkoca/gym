import matplotlib
matplotlib.use('Agg')
import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def draw_frame(info, fitness_history, all_input_ids=None, all_output_ids=None, figsize=(16, 6), fallback_shape=None):
    # --- Filter to only active nodes/connections (on a path from input to output via enabled connections) ---
    nodes = info['nodes']
    connections = info['connections']
    node_types = {n['id']: n['type'] for n in nodes}
    input_ids = [n['id'] for n in nodes if n['type'] == 'input']
    output_ids = [n['id'] for n in nodes if n['type'] == 'output']
    node_id_set = set(n['id'] for n in nodes)
    # Build adjacency list for enabled connections, only if both nodes exist
    from collections import deque
    adj = {n['id']: [] for n in nodes}
    for c in connections:
        if c['enabled'] and c['in_node'] in node_id_set and c['out_node'] in node_id_set:
            adj[c['in_node']].append(c['out_node'])
    # Forward BFS from inputs to find reachable nodes
    reachable = set()
    queue = deque(input_ids)
    while queue:
        nid = queue.popleft()
        if nid in reachable:
            continue
        reachable.add(nid)
        for neighbor in adj.get(nid, []):
            if neighbor not in reachable:
                queue.append(neighbor)
    # Backward BFS from outputs to find nodes that can reach outputs
    rev_adj = {n['id']: [] for n in nodes}
    for c in connections:
        if c['enabled'] and c['in_node'] in node_id_set and c['out_node'] in node_id_set:
            rev_adj[c['out_node']].append(c['in_node'])
    can_reach_output = set()
    queue = deque(output_ids)
    while queue:
        nid = queue.popleft()
        if nid in can_reach_output:
            continue
        can_reach_output.add(nid)
        for neighbor in rev_adj.get(nid, []):
            if neighbor not in can_reach_output:
                queue.append(neighbor)
    # Only keep nodes that are both reachable from input and can reach output
    active_nodes = reachable & can_reach_output
    # Only keep connections where both nodes are active and connection is enabled
    active_conns = [c for c in connections if c['enabled'] and c['in_node'] in active_nodes and c['out_node'] in active_nodes]
    # Only keep nodes that are active
    active_nodes_list = [n for n in nodes if n['id'] in active_nodes]
    # Replace info for plotting
    info = dict(info)
    info['nodes'] = active_nodes_list
    info['connections'] = active_conns

    logging.info(f"Drawing frame for generation {info.get('generation', '?')}")
    G = nx.DiGraph()
    label_map = {}
    node_types = {}
    for node in info['nodes']:
        G.add_node(node['id'])
        label_map[node['id']] = f"{node['type']}\n{node['id']}"
        node_types[node['id']] = node['type']
    color_map_dict = {nid: ('skyblue' if node_types[nid] == 'input' else
                            'lightgreen' if node_types[nid] == 'output' else
                            'orange') for nid in G.nodes()}
    for conn in info['connections']:
        if conn['enabled']:
            G.add_edge(conn['in_node'], conn['out_node'], weight=conn['weight'])
    input_nodes = [n['id'] for n in info['nodes'] if n['type'] == 'input']
    output_nodes = [n['id'] for n in info['nodes'] if n['type'] == 'output']
    hidden_nodes = [n['id'] for n in info['nodes'] if n['type'] == 'hidden']
    # --- Compute topological depth for each node (for x-position) ---
    from collections import deque, defaultdict
    G_enabled = nx.DiGraph()
    for node in G.nodes():
        G_enabled.add_node(node)
    for conn in info['connections']:
        if conn['enabled']:
            G_enabled.add_edge(conn['in_node'], conn['out_node'])
    node_depth = {nid: 0 for nid in input_nodes}
    queue = deque(input_nodes)
    bfs_steps = 0
    bfs_limit = 10000
    while queue:
        nid = queue.popleft()
        bfs_steps += 1
        if bfs_steps > bfs_limit:
            logging.warning(f"BFS for topological depth exceeded {bfs_limit} steps at node {nid}, skipping frame.")
            if fallback_shape is not None:
                return np.zeros(fallback_shape, dtype=np.uint8)
            else:
                return np.zeros((100, 100, 3), dtype=np.uint8)
        for _, v in G_enabled.out_edges(nid):
            if v not in node_depth or node_depth[v] < node_depth[nid] + 1:
                node_depth[v] = node_depth[nid] + 1
                queue.append(v)
    max_depth = max([node_depth.get(nid, 0) for nid in output_nodes] + [1])
    pos = {}
    y_by_depth = defaultdict(list)
    for nid in node_depth:
        y_by_depth[node_depth[nid]].append(nid)
    for depth, nids in y_by_depth.items():
        n = len(nids)
        for i, nid in enumerate(sorted(nids)):
            pos[nid] = (depth, 0.5 - i * (1.0/(n-1)) if n > 1 else 0)
    for nid in output_nodes:
        if nid not in pos:
            pos[nid] = (max_depth, 0)
    for nid in input_nodes:
        if nid not in pos:
            pos[nid] = (0, 0)
    # Fix: Always space input and output nodes vertically, using all possible input/output node ids from the full progress history
    # Find all input/output node ids from the full progress (not just current info)
    if all_input_ids is None:
        all_input_ids = sorted([n['id'] for n in info['nodes'] if n['type'] == 'input'])
    if all_output_ids is None:
        all_output_ids = sorted([n['id'] for n in info['nodes'] if n['type'] == 'output'])
    # Place input nodes at x=0, spaced vertically
    for i, nid in enumerate(all_input_ids):
        pos[nid] = (0, 1.0 - i * (2.0/(max(1, len(all_input_ids)-1))) if len(all_input_ids) > 1 else 0)
    # Place output nodes at x=max_depth, spaced vertically
    for i, nid in enumerate(all_output_ids):
        pos[nid] = (max_depth, 1.0 - i * (2.0/(max(1, len(all_output_ids)-1))) if len(all_output_ids) > 1 else 0)
    # Plot
    fig = plt.figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
    ax0 = fig.add_subplot(gs[0])
    # Only draw edges where both nodes are in pos
    edges_to_draw = [(u, v) for u, v in G.edges() if u in pos and v in pos]
    G_sub = G.edge_subgraph(edges_to_draw).copy()
    nodes_to_draw = [n for n in G_sub.nodes if n in pos]
    color_map_draw = [
        'skyblue' if node_types.get(n) == 'input'
        else 'lightgreen' if node_types.get(n) == 'output'
        else 'orange'
        for n in nodes_to_draw
    ]
    label_map_draw = {n: label_map[n] if n in label_map else str(n) for n in nodes_to_draw}
    # --- Strictly filter to only nodes/edges on a path from input to output via enabled connections ---
    # Build a subgraph of only enabled edges
    G_enabled = nx.DiGraph()
    for node in G.nodes():
        G_enabled.add_node(node)
    for conn in info['connections']:
        if conn['enabled']:
            G_enabled.add_edge(conn['in_node'], conn['out_node'], weight=conn['weight'])
    # Forward: nodes reachable from any input
    reachable_from_input = set()
    for start in input_nodes:
        stack = [start]
        steps = 0
        while stack:
            node = stack.pop()
            steps += 1
            if steps > bfs_limit:
                logging.warning(f"Forward traversal exceeded {bfs_limit} steps at node {node}, skipping frame.")
                return np.zeros(fallback_shape or (100, 100, 3), dtype=np.uint8)
            if node not in reachable_from_input:
                reachable_from_input.add(node)
                stack.extend([v for _, v in G_enabled.out_edges(node)])
    # Backward: nodes that can reach any output
    can_reach_output = set()
    for end in output_nodes:
        stack = [end]
        steps = 0
        while stack:
            node = stack.pop()
            steps += 1
            if steps > bfs_limit:
                logging.warning(f"Backward traversal exceeded {bfs_limit} steps at node {node}, skipping frame.")
                return np.zeros(fallback_shape or (100, 100, 3), dtype=np.uint8)
            if node not in can_reach_output:
                can_reach_output.add(node)
                stack.extend([u for u, _ in G_enabled.in_edges(node)])
    # Only keep nodes/edges in both sets
    active_nodes = reachable_from_input & can_reach_output
    active_edges = [(u, v) for u, v in G_enabled.edges() if u in active_nodes and v in active_nodes]
    G_sub = G_enabled.edge_subgraph(active_edges).copy()
    nodes_to_draw = [n for n in G_sub.nodes if n in pos]
    color_map_draw = [color_map_dict[n] for n in nodes_to_draw]
    label_map_draw = {n: label_map[n] if n in label_map else str(n) for n in nodes_to_draw}
    nx.draw(G_sub, pos, nodelist=nodes_to_draw, with_labels=True, labels=label_map_draw, node_color=color_map_draw, node_size=800, arrows=True, ax=ax0, font_size=8)
    edge_labels = {(c['in_node'], c['out_node']): f"{c['weight']:.2f}" for c in info['connections'] if c['enabled'] and c['in_node'] in active_nodes and c['out_node'] in active_nodes and c['in_node'] in pos and c['out_node'] in pos}
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=7, ax=ax0)
    ax0.set_title("Network Structure", fontsize=12)
    ax0.axis('off')
    # Fitness curve
    ax1 = fig.add_subplot(gs[1])
    gens = [f['generation'] for f in fitness_history]
    fits = [f['best_fitness'] for f in fitness_history]
    ax1.plot(gens, fits, color='tab:blue', marker='o', markersize=3, linewidth=1)
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Fitness Progress')
    ax1.axvline(info['generation'], color='red', linestyle='--', alpha=0.5)
    fig.suptitle(f"Best Genome - Gen {info['generation']} | Fitness: {info['best_fitness']:.3f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Convert to image array using Agg canvas
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    img = img[..., :3]  # Drop alpha channel for RGB
    plt.close(fig)
    return img

def make_gif_from_metadata(
    metadata_path="output/progress.json",
    gif_path="output/progress.gif",
    duration_ms=60000
):
    import imageio
    with open(metadata_path) as f:
        progress = json.load(f)
    all_input_ids = sorted({n['id'] for info in progress for n in info['nodes'] if n['type'] == 'input'})
    all_output_ids = sorted({n['id'] for info in progress for n in info['nodes'] if n['type'] == 'output'})
    images = []
    last_shape = None
    for i, info in enumerate(progress):
        logging.info(f"Generating frame {i+1}/{len(progress)} (generation {info.get('generation', '?')})")
        frame = draw_frame(info, progress[:i+1], all_input_ids=all_input_ids, all_output_ids=all_output_ids, fallback_shape=last_shape)
        if frame is not None:
            last_shape = frame.shape
            images.append(frame)
    n_frames = len(images)
    if n_frames == 0:
        print("No frames to save!")
        return
    duration_sec = duration_ms / 1000.0
    per_frame_duration = duration_sec / n_frames
    imageio.mimsave(gif_path, images, duration=per_frame_duration)
    print(f"GIF saved to {gif_path} (duration per frame: {per_frame_duration:.3f}s, total: {duration_sec:.2f}s, frames: {n_frames})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a GIF from NEAT progress metadata.")
    parser.add_argument('--metadata_path', type=str, default="output/progress.json", help='Path to progress.json')
    parser.add_argument('--gif_path', type=str, default="output/progress.gif", help='Path to save GIF')
    parser.add_argument('--duration_ms', type=int, default=60000, help='Frame duration in ms (default: 60000)')
    args = parser.parse_args()
    make_gif_from_metadata(args.metadata_path, args.gif_path, args.duration_ms)
