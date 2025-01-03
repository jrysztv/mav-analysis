# %%
# import pandas as pd
from igraph import Graph
from config import dir_config
import pandas as pd

from mav_analysis.grapher.utils.graph_builder import StopSequenceGraphBuilder
from mav_analysis.grapher.utils.graph_visualizer import MapVisualizer

# %% Load data
stops_df = pd.read_csv(dir_config.mav_gtfs_dir / "stops.txt")
stop_times_df = pd.read_csv(dir_config.mav_gtfs_dir / "stop_times.txt")
stop_lines_df = pd.read_parquet(
    dir_config.input_graph_data_dir / "stop_lines_df.parquet"
)
stop_lines_df_uninterpolated = pd.read_parquet(
    dir_config.input_graph_data_dir / "stop_lines_uninterpolate_df.parquet"
)

# %%
gtfs_graph_handler = StopSequenceGraphBuilder(
    stops_df, stop_times_df, group_by="trip_id"
)
gtfs_graph_handler.run()

gtfs_map_visualizer = MapVisualizer(gtfs_graph_handler.graph)
gtfs_map_visualizer.save_folium_map("gtfs_map.html")

# %%
line_shape_graph_handler = StopSequenceGraphBuilder(
    stops_df, stop_lines_df, group_by="elvira_id"
)
line_shape_graph_handler.run()

line_shape_map_visualizer = MapVisualizer(line_shape_graph_handler.graph)
line_shape_map_visualizer.save_folium_map("line_shape_map.html")

# %%
line_shape_graph_handler_uninterpolated = StopSequenceGraphBuilder(
    stops_df, stop_lines_df_uninterpolated, group_by="elvira_id"
)
line_shape_graph_handler_uninterpolated.run()

line_shape_map_visualizer_uninterpolated = MapVisualizer(
    line_shape_graph_handler_uninterpolated.graph
)
line_shape_map_visualizer_uninterpolated.save_folium_map(
    "line_shape_map_uninterpolated.html"
)
# %% Comparing graphs
line_shape_graph = line_shape_graph_handler.graph
line_shape_graph_undirected = line_shape_graph.as_undirected(mode="collapse")

gtfs_graph = gtfs_graph_handler.graph
gtfs_graph_undirected = gtfs_graph.as_undirected(mode="collapse")


# %%
stop_ids_line_shape = set(line_shape_graph_undirected.vs["stop_id"])
stop_ids_gtfs = set(gtfs_graph_undirected.vs["stop_id"])
# combine
all_stop_ids = sorted(set(stop_ids_line_shape).union(set(stop_ids_gtfs)))

stopid_to_newidx = {stop_id: idx for idx, stop_id in enumerate(all_stop_ids)}


def refactor_graph(old_graph, stopid_to_newidx, all_stop_ids):
    """
    Build a new graph with vertex IDs aligned to stop_id -> new_index.
    Copy over vertex and edge attributes as needed.
    """
    # 1) Create new graph with the full set of vertices
    new_n = len(all_stop_ids)
    new_graph = Graph(n=new_n, directed=old_graph.is_directed())

    # 2) Copy the vertex attributes
    # For example, let's store 'stop_id' in the new graph
    new_graph.vs["stop_id"] = (
        all_stop_ids  # each new vertex i has stop_id = all_stop_ids[i]
    )

    # 3) Remap edges
    # We can also copy attributes from old_graph edges
    edges_to_add = []
    edge_attrs = {}  # dict of attribute_name -> list of attribute_values

    # Prepare lists for edge attributes if you want to copy them
    for attr_name in old_graph.edge_attributes():
        edge_attrs[attr_name] = []

    for e in old_graph.es:
        old_u, old_v = e.tuple
        sid_u = old_graph.vs[old_u]["stop_id"]
        sid_v = old_graph.vs[old_v]["stop_id"]
        new_u = stopid_to_newidx[sid_u]
        new_v = stopid_to_newidx[sid_v]

        edges_to_add.append((new_u, new_v))

        # Copy attributes
        for attr_name in edge_attrs:
            val = e[attr_name]
            edge_attrs[attr_name].append(val)

    # 4) Add edges
    new_graph.add_edges(edges_to_add)

    # 5) Attach edge attributes
    for attr_name in edge_attrs:
        new_graph.es[attr_name] = edge_attrs[attr_name]

    return new_graph


line_shape_graph_undirected = refactor_graph(
    line_shape_graph_undirected, stopid_to_newidx, all_stop_ids
)

gtfs_graph_undirected = refactor_graph(
    gtfs_graph_undirected, stopid_to_newidx, all_stop_ids
)


# Merge stop_name, stop_lat, stop_lon based on stop_id from stops_df
def populate_stop_attributes(stops_df, graph):
    graph.vs["stop_name"] = [
        stops_df.loc[stops_df["stop_id"] == stop_id, "stop_name"].values[0]
        for stop_id in graph.vs["stop_id"]
    ]
    graph.vs["stop_lat"] = [
        stops_df.loc[stops_df["stop_id"] == stop_id, "stop_lat"].values[0]
        for stop_id in graph.vs["stop_id"]
    ]
    graph.vs["stop_lon"] = [
        stops_df.loc[stops_df["stop_id"] == stop_id, "stop_lon"].values[0]
        for stop_id in graph.vs["stop_id"]
    ]
    return graph


line_shape_graph_undirected = populate_stop_attributes(
    stops_df, line_shape_graph_undirected
)

gtfs_graph_undirected = populate_stop_attributes(stops_df, gtfs_graph_undirected)
# %%


# %%
# 2) Extract edge lists and convert to sets of sorted tuples
def edge_set(graph):
    return set(tuple(sorted(e)) for e in graph.get_edgelist())


edges_line_shape = edge_set(line_shape_graph_undirected)
edges_gtfs = edge_set(gtfs_graph_undirected)

common_edges = edges_line_shape.intersection(edges_gtfs)

to_remove_gtfs = []
for e in gtfs_graph_undirected.es:
    # e.tuple returns the endpoints (u,v); sort so undirected edges match sorted tuples
    sorted_tuple = tuple(sorted(e.tuple))
    if sorted_tuple in common_edges:
        to_remove_gtfs.append(e.index)

gtfs_graph_complementary = gtfs_graph_undirected.copy()

gtfs_graph_complementary.delete_edges(to_remove_gtfs)

to_remove_line_shape = []
for e in line_shape_graph_undirected.es:
    sorted_tuple = tuple(sorted(e.tuple))
    if sorted_tuple in common_edges:
        to_remove_line_shape.append(e.index)


line_shape_graph_complementary = line_shape_graph_undirected.copy()

# Actually remove them
line_shape_graph_complementary.delete_edges(to_remove_line_shape)

line_shape_complementary_visualizer = MapVisualizer(
    line_shape_graph_complementary, directed=False
)
line_shape_complementary_visualizer.save_folium_map("line_shape_complementary.html")

gtfs_complementary_visualizer = MapVisualizer(gtfs_graph_complementary, directed=False)
gtfs_complementary_visualizer.save_folium_map("gtfs_complementary.html")
