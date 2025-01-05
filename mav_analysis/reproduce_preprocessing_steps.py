from .align_schedule_to_graph import align_schedule_to_graph
from .gather_input_graph_data import gather_and_save_graph_building_inputs
from .feature_table_finalization import construct_daily_graph_features_data
from .download_mav_scrape_data import download_and_save_all_scraped_mav_data


def reproduce_preprocessing_steps():
    """
    Execute the preprocessing steps for the MAV analysis project.

    This function performs the following steps:
    1. Downloads and saves all scraped MAV data.
    2. Gathers and saves graph building inputs.
    3. Aligns the schedule to the graph.
    4. Creates daily feature tables.

    After running this function, the preprocessing steps will be completed,
    and the analysis can be run in the analysis.ipynb notebook in the project root.
    """
    print("Starting preprocessing steps...")
    download_and_save_all_scraped_mav_data()
    print("Scraped data downloaded and saved.")
    print("Gathering and saving graph building inputs...")
    gather_and_save_graph_building_inputs()
    print("Graph building inputs gathered and saved.")
    print("Aligning schedule to graph...")
    align_schedule_to_graph()
    print("Schedule aligned to graph.")
    print("Creating daily feature tables...")
    construct_daily_graph_features_data()
    print("Daily feature tables created.")
    print("Preprocessing steps completed successfully.")
    print("You can now proceed to run the analysis in analysis.ipynb in project root.")
    print("Have a nice day!")
