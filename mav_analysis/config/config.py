from pydantic import BaseModel
from pathlib import Path


class DirConfig(BaseModel):
    # data directories
    data_dir: Path
    input_graph_data_dir: Path
    mav_gtfs_dir: Path
    working_dir: Path
    mav_scrape_dir: Path
    # results directories
    results_dir: Path
