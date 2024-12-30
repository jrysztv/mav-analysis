# %%
import pandas as pd
from igraph import Graph
import folium
from physical_routes_fetcher.fetch_shape_routes import gather_stops_for_lines
from physical_routes_fetcher.utils.shapes_block_request import ShapesRequest


class GraphHandler:
    def __init__(self, stops_df, stop_sequence_df, group_by="trip_id"):
        self.stops_df = stops_df
        self.stop_sequence_df = stop_sequence_df
        self.group_by_col = group_by
        self.vertex_attributes = None
        self.edge_df = None
        self.graph = None

    def add_vertice_id(self):
        self.stop_sequence_df["vertice_id"] = (
            self.stop_sequence_df["stop_id"].astype("category").cat.codes
        )

    def add_prev_vertice_id(self):
        self.stop_sequence_df["prev_vertice_id"] = (
            self.stop_sequence_df.groupby(self.group_by_col)["vertice_id"]
            .shift(1)
            .astype("Int64")
        )

    def create_edge_df(self):
        self.edge_df = (
            self.stop_sequence_df[["prev_vertice_id", "vertice_id"]]
            .dropna()
            .drop_duplicates()
        )

    def create_vertex_attributes(self):
        vertex_attributes = (
            self.stop_sequence_df[["vertice_id", "stop_id"]]
            .drop_duplicates()
            .set_index("stop_id")
        )
        vertex_attributes["stop_name"] = self.stops_df.set_index("stop_id")["stop_name"]
        vertex_attributes["stop_lat"] = self.stops_df.set_index("stop_id")["stop_lat"]
        vertex_attributes["stop_lon"] = self.stops_df.set_index("stop_id")["stop_lon"]
        self.vertex_attributes = (
            vertex_attributes.reset_index().set_index("vertice_id").sort_index()
        )

    def create_graph(self):
        self.graph = Graph(
            vertex_attrs=self.vertex_attributes.to_dict("list"),
            edges=self.edge_df[["vertice_id", "prev_vertice_id"]].values.tolist(),
            directed=True,
        )

    def run(self):
        self.add_vertice_id()
        self.add_prev_vertice_id()
        self.create_edge_df()
        self.create_vertex_attributes()
        self.create_graph()


class MapVisualizer:
    def __init__(self, graph, shapes_block_request=None):
        self.graph = graph
        self.shapes_block_request = shapes_block_request

    def get_map_center(self):
        lats = [v["stop_lat"] for v in self.graph.vs]
        lons = [v["stop_lon"] for v in self.graph.vs]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def calculate_edge_length(self, lat1, lon1, lat2, lon2):
        from geopy.distance import geodesic

        return geodesic((lat1, lon1), (lat2, lon2)).km

    def get_edge_color(self, length):
        if length >= 20:
            return "red"
        elif length >= 10:
            return "yellow"
        else:
            return "blue"

    def plot_shapes(self, m):
        if self.shapes_block_request is not None:
            for shape in self.shapes_block_request:
                folium.PolyLine(
                    locations=shape["coordinates"],
                    color=shape.get("color", "green"),
                    weight=shape.get("weight", 2),
                    opacity=shape.get("opacity", 0.6),
                ).add_to(m)

    def create_folium_map(self):
        center_lat, center_lon = self.get_map_center()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        for vertex in self.graph.vs:
            lat, lon = vertex["stop_lat"], vertex["stop_lon"]
            tooltip_text = vertex["stop_name"]
            folium.CircleMarker(
                [lat, lon], tooltip=tooltip_text, radius=3, weight=5, color="#006dfc"
            ).add_to(m)

        for edge in self.graph.es:
            source = edge.source
            target = edge.target
            lat1, lon1 = (
                self.graph.vs[source]["stop_lat"],
                self.graph.vs[source]["stop_lon"],
            )
            lat2, lon2 = (
                self.graph.vs[target]["stop_lat"],
                self.graph.vs[target]["stop_lon"],
            )

            length = self.calculate_edge_length(lat1, lon1, lat2, lon2)
            color = self.get_edge_color(length)

            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=color,
                weight=3,
                opacity=0.6,
            ).add_to(m)

        self.plot_shapes(m)

        return m


# 1) Load stops
stops_df = pd.read_csv("data/mav_gtfs/stops.txt")

# 2) Load stop_times
stop_times_df = pd.read_csv("data/mav_gtfs/stop_times.txt")

# 3) Gather stops for lines for comparison
stop_lines_block_df = gather_stops_for_lines(1000)

# 4) Request shapes to compare the graph with the initial shapes
shapes_request = ShapesRequest()
shape_df = shapes_request.request_and_extract()

# Convert shape_df to shapes_block_request format
shapes_block_request = []
for line_id, group in shape_df.groupby("line_id"):
    coordinates = group[["lat", "lon"]].values.tolist()
    shapes_block_request.append(
        {"coordinates": coordinates, "color": "purple", "weight": 5}
    )

# 5) load stop_lines_by_train
stop_lines_by_train_df = gather_stops_for_lines(
    by_train_collection=True, group_id="elvira_id"
)


# %%
# Process data
gtfs_graph_handler = GraphHandler(stops_df, stop_times_df, group_by="trip_id")
gtfs_graph_handler.run()

# Create and display map with shapes
gtfs_map_visualizer = MapVisualizer(gtfs_graph_handler.graph)
gtfs_map_plot = gtfs_map_visualizer.create_folium_map()
gtfs_map_plot
# %%
line_shape_graph_handler = GraphHandler(
    stops_df, stop_lines_block_df, group_by="line_id"
)
line_shape_graph_handler.run()

line_shape_map_visualizer = MapVisualizer(
    line_shape_graph_handler.graph, shapes_block_request=shapes_block_request
)
line_shape_map_plot = line_shape_map_visualizer.create_folium_map()
line_shape_map_plot
# %%
line_shape_graph_handler = GraphHandler(
    stops_df, stop_lines_by_train_df, group_by="elvira_id"
)
line_shape_graph_handler.run()
line_shape_map_visualizer = MapVisualizer(line_shape_graph_handler.graph)
line_shape_map_plot = line_shape_map_visualizer.create_folium_map()
line_shape_map_plot

# %%
stop_lines_by_train_df
# %%
parsed_train_shapes = pd.read_parquet("data/parsed_train_shapes.parquet")
# %%
parsed_train_shapes
# %%
