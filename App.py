"""
Interactive Search Visualization (Streamlit)
--------------------------------------------
Run with:   streamlit run streamlit_app.py

Uses the same weighted graphs and the same GBFS / A* implementations as the
notebook (Tasks 1-4). Pick a graph, a start node, a goal node and an algorithm,
then press "Run search" to see the solution path highlighted on the graph.
"""
import heapq
import itertools
import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Graph definitions (same data as the notebook tasks)
# ----------------------------------------------------------------------------
GRAPHS = {
    "Hospital (Task 3)": {
        "coords": {
            "Pharmacy": (0, 0), "Main_Corridor": (2, 1), "Patient_Wing": (1, 4),
            "Nursing_Station": (4, 2), "Laboratory": (5, 5), "Emergency_Ward": (8, 6),
        },
        "edges": [
            ("Pharmacy", "Patient_Wing", 4.1), ("Pharmacy", "Main_Corridor", 2.2),
            ("Main_Corridor", "Nursing_Station", 2.2), ("Patient_Wing", "Laboratory", 5.0),
            ("Nursing_Station", "Laboratory", 3.2), ("Nursing_Station", "Emergency_Ward", 6.0),
            ("Laboratory", "Emergency_Ward", 3.2),
        ],
        "default": ("Pharmacy", "Emergency_Ward"),
    },
    "Airport (Task 2)": {
        "coords": {
            "Baggage_Area": (0, 0), "Security": (2, 1), "Checkpoint": (1, 4),
            "Food_Court": (4, 2), "Terminal_Hall": (5, 5), "Departure_Gate": (8, 6),
        },
        "edges": [
            ("Baggage_Area", "Checkpoint", 4.1), ("Baggage_Area", "Security", 2.2),
            ("Security", "Food_Court", 2.2), ("Checkpoint", "Terminal_Hall", 5.0),
            ("Food_Court", "Terminal_Hall", 3.2), ("Food_Court", "Departure_Gate", 6.0),
            ("Terminal_Hall", "Departure_Gate", 3.2),
        ],
        "default": ("Baggage_Area", "Departure_Gate"),
    },
    "Drone delivery (Task 4)": {
        "coords": {
            "Distribution_Center": (0, 0), "Zone_A": (2, 1), "Zone_B": (1, 4),
            "Zone_C": (4, 2), "Zone_D": (5, 5), "Customer_Building": (8, 6),
        },
        "edges": [
            ("Distribution_Center", "Zone_B", 3.0), ("Distribution_Center", "Zone_A", 2.2),
            ("Zone_A", "Zone_C", 2.2), ("Zone_B", "Zone_D", 5.0),
            ("Zone_C", "Zone_D", 3.2), ("Zone_C", "Customer_Building", 6.0),
            ("Zone_D", "Customer_Building", 3.2),
        ],
        "default": ("Distribution_Center", "Customer_Building"),
    },
    "Warehouse (Task 1)": {
        "coords": {
            "Receiving_Area": (0, 0), "Storage_A": (2, 1), "Storage_B": (1, 4),
            "Sorting_Area": (4, 2), "Inspection_Area": (5, 5), "Packing_Station": (7, 6),
        },
        "edges": [
            ("Receiving_Area", "Storage_B", 4.1), ("Receiving_Area", "Storage_A", 2.2),
            ("Storage_A", "Sorting_Area", 2.2), ("Storage_B", "Inspection_Area", 5.0),
            ("Storage_B", "Sorting_Area", 6.0), ("Sorting_Area", "Inspection_Area", 3.2),
            ("Sorting_Area", "Packing_Station", 5.0), ("Inspection_Area", "Packing_Station", 2.2),
        ],
        "default": ("Receiving_Area", "Packing_Station"),
    },
}


def build_graph(coords, edges):
    G = nx.DiGraph()
    for n, p in coords.items():
        G.add_node(n, pos=p)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


def euclid(coords, a, b):
    return math.dist(coords[a], coords[b])


def path_cost(G, path):
    return sum(G[u][v]["weight"] for u, v in zip(path, path[1:]))


def _rebuild(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ----------------------------------------------------------------------------
# Search algorithms (same logic as the notebook)
# ----------------------------------------------------------------------------
def gbfs(G, coords, start, goal):
    """Greedy Best-First Search: f(n) = h(n)."""
    h = lambda n: euclid(coords, n, goal)
    counter = itertools.count()
    frontier = [(h(start), next(counter), start)]
    parent, g = {start: None}, {start: 0.0}
    explored, order, rows = set(), [], []
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        order.append(node)
        rows.append({"Node": node, "g(n)": round(g[node], 3), "h(n)": round(h(node), 3),
                     "f(n) = h(n)": round(h(node), 3)})
        if node == goal:
            return _rebuild(parent, node), order, rows
        for nb in G.successors(node):
            if nb not in explored and nb not in parent:
                parent[nb] = node
                g[nb] = g[node] + G[node][nb]["weight"]
                heapq.heappush(frontier, (h(nb), next(counter), nb))
    return None, order, rows


def astar(G, coords, start, goal):
    """A* Search: f(n) = g(n) + h(n)."""
    h = lambda n: euclid(coords, n, goal)
    counter = itertools.count()
    g, parent = {start: 0.0}, {start: None}
    frontier = [(h(start), next(counter), start)]
    order, rows = [], []
    while frontier:
        f, _, node = heapq.heappop(frontier)
        if f > g[node] + h(node) + 1e-12:      # stale queue entry
            continue
        order.append(node)
        rows.append({"Node": node, "g(n)": round(g[node], 3), "h(n)": round(h(node), 3),
                     "f(n) = g + h": round(g[node] + h(node), 3)})
        if node == goal:
            return _rebuild(parent, node), order, rows
        for nb in G.successors(node):
            new_g = g[node] + G[node][nb]["weight"]
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                parent[nb] = node
                heapq.heappush(frontier, (new_g + h(nb), next(counter), nb))
    return None, order, rows


ALGORITHMS = {"GBFS": gbfs, "A*": astar}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw(G, coords, start, goal, path):
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(figsize=(10, 6))
    on_path = set(zip(path, path[1:])) if path else set()
    colors = []
    for n in G.nodes:
        if n == start:
            colors.append("#f28b82")
        elif n == goal:
            colors.append("#fdd663")
        elif path and n in path:
            colors.append("#a8dab5")
        else:
            colors.append("#cfd8ea")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1700, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges if e not in on_path],
                           arrows=True, arrowsize=18, node_size=1700, width=1.3, ax=ax)
    if on_path:
        nx.draw_networkx_edges(G, pos, edgelist=list(on_path), arrows=True, arrowsize=22,
                               node_size=1700, width=4, edge_color="crimson", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=9,
                                 edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)})
    nx.draw_networkx_labels(G, {n: (x, y - 0.65) for n, (x, y) in pos.items()},
                            {n: f"{n}\n{coords[n]}" for n in G.nodes}, font_size=8, ax=ax)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.8, max(ys) + 1.5)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Search Visualizer", layout="wide")
st.title("Search Algorithm Visualizer")
st.caption("Greedy Best-First Search and A* on a weighted graph, heuristic = Euclidean distance to the goal.")

graph_name = st.sidebar.selectbox("Graph", list(GRAPHS.keys()))
cfg = GRAPHS[graph_name]
coords = cfg["coords"]
G = build_graph(coords, cfg["edges"])
nodes = list(coords.keys())
def_start, def_goal = cfg["default"]

c1, c2, c3 = st.columns(3)
start = c1.selectbox("Initial node", nodes, index=nodes.index(def_start), key=f"s_{graph_name}")
goal = c2.selectbox("Goal node", nodes, index=nodes.index(def_goal), key=f"g_{graph_name}")
algo = c3.selectbox("Search algorithm", list(ALGORITHMS.keys()), index=1)

if st.button("Run search", type="primary"):
    if start == goal:
        st.session_state["result"] = {"error": "Initial node and goal node must be different."}
    else:
        path, order, rows = ALGORITHMS[algo](G, coords, start, goal)
        st.session_state["result"] = {
            "graph": graph_name, "start": start, "goal": goal, "algo": algo,
            "path": path, "order": order, "rows": rows,
        }

res = st.session_state.get("result")
if res and res.get("graph", graph_name) != graph_name and "error" not in res:
    res = None                     # a result from another graph is not shown

if not res:
    st.subheader("Graph")
    st.pyplot(draw(G, coords, start, goal, None))
    st.info("Choose the nodes and algorithm, then press **Run search**.")
elif "error" in res:
    st.error(res["error"])
else:
    st.subheader("Graph")
    st.pyplot(draw(G, coords, res["start"], res["goal"], res["path"]))
    st.subheader("Result")
    if res["path"] is None:
        st.error(f"No path from {res['start']} to {res['goal']}.")
    else:
        st.markdown(f"**Selected algorithm:** {res['algo']}")
        st.markdown("**Solution path:** " + " → ".join(res["path"]))
        st.markdown(f"**Total path cost:** {path_cost(G, res['path']):.2f}")
    st.markdown("**Node expansion order:** " + " → ".join(res["order"]))
    st.markdown("**Expanded nodes: g(n), h(n), f(n)**")
    st.dataframe(pd.DataFrame(res["rows"]), hide_index=True)
