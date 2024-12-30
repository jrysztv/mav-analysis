# %%
import pandas as pd
import numpy as np
from physical_routes_fetcher.utils.shapes_route_assign import RouteStopAnalyzer
from physical_routes_fetcher.utils.shapes_block_request import ShapesRequest


# %%
def gather_stops_for_lines(
    distance_threshold_meters=1000,
    by_train_collection: bool = False,
    train_collection_path="data/parsed_train_shapes.parquet",
    group_id="line_id",
):
    if by_train_collection:
        route_df = pd.read_parquet(train_collection_path)
        route_df = route_df.rename(columns={"lat": "stop_lat", "lon": "stop_lon"})
    else:
        shapes_requester = ShapesRequest()

        route_df = shapes_requester.request_and_extract().rename(
            columns={"lat": "stop_lat", "lon": "stop_lon"}
        )

    stops_df = pd.read_csv("data/mav_gtfs/stops.txt")

    analyzer = RouteStopAnalyzer(
        route_df,
        stops_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        stop_id_col="stop_id",
    )

    grouped_encountered_stops_df = analyzer.stop_sequence_by_shape_grouped(
        group_id=group_id,
        distance_threshold_meters=distance_threshold_meters,
        buffer_degrees=0.005,
    )

    return grouped_encountered_stops_df.merge(stops_df.dropna(axis=1), on="stop_id")


if __name__ == "__main__":
    df = gather_stops_for_lines(1000)
