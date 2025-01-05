# MAV Requests Analysis

## Project Setup

This project is managed using [Poetry](https://python-poetry.org/). Follow the steps below to set it up:

1. **Install Poetry**:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Install Dependencies**:
    ```sh
    poetry install
    ```

3. **Activate the Virtual Environment**:
    ```sh
    poetry shell
    ```

## Analysis

The analysis is available in the root folder in the `analysis.ipynb` notebook.

## Reproducible Preprocessing Pipeline

To reproduce the preprocessing steps, run the following script:
```sh
./mav_analysis/reproduce_preprocessing_steps.py
```

## Data Scraping

Data is scraped from a public API intercepted from the network activity here: [https://vonatinfo.mav-start.hu/](https://vonatinfo.mav-start.hu/). Some details on how the network traffic was intercepted, and how to automate such jobs on GitHub are explained here: [https://medium.com/p/25cc2111cb3c](https://medium.com/p/25cc2111cb3c). The implemented scraping logic can be explored here: [https://github.com/jrysztv/MAV-scraper](https://github.com/jrysztv/MAV-scraper). The scrape is intended to run every 5 minutes, but in reality, it runs between every 12-30 minutes.

## Source of the Data

- **Current locations and delays of the trains at the time of request**, and the list of trains that are currently traveling.
- **The schedule of a train** with a list of stop names, as well as arrival and departure times (hours:minutes).
- The website lists updated times in red below these scheduled times. If a train passes a station, this time remains there, and is fixed.
- At each scraping epoch, we update the train's stored schedule (scheduled and estimated arrival and departure times by station), and this goes on, until it travels. The last table is the full schedule with scheduled and materialized arrival times. The difference between the two are the delays.
- **The line a train takes on its entire route**. These are called shapes in GTFS standard.
- Shapes are a series of coordinates in a fixed order. Interpolating them yields a close approximation of the physical route a train took.

## Transforming the Raw Data into the Graph Dataset

Upon request and a short registration, the Hungarian State Railways (MÁV) grants access to their General Transit Feed Specification (GTFS) data here: [https://www.mavcsoport.hu/gtfs-igenybejelento](https://www.mavcsoport.hu/gtfs-igenybejelento).

- GTFS is a standardized data format accepted globally for sharing schedule data.
- We use the list of stops from this dataset, from `stops.txt`.
- The GTFS data also contains information on the schedule of each train. However, these are difficult to use:
  - They cannot be connected to the scraped information of actual train data.
  - If we were to build a graph from the adjacency of sequential stops, InterCity trains for example show up as if they were skipping a lot of stops, creating phantom edges.
- Shapes in combination with stops are a lot better for these two reasons:
  - We instead build the full directed graph based on the location of stops, and the exact route of all trains. Wherever a train passes by, the edge is created.
  - We create daily weighted graphs based on the number of trains passing on an edge as a weight.
  - Other features, such as distance, time, and speed are also added as different weight categories used to aggregate node-neighborhood measures.

## Describing the Dataset

### ID Variables

- `vertex_id` - unique for each stop.
- `date` - unique for each day.

### Variables in the Baseline Models

- **Dependent Variable**: `avg_incoming_delay`
  - This measures the sum of daily delays in minutes divided by the sum of incoming trains.
  - About 60% of observations are non-missing. This will be the bottleneck in terms of the number of observations in the analysis.
- **Explanatory Variable**: `weighted_betweenness`
  - This gauges the general importance of a node in the network and accounts for the number of trains transiting.

### Control Variables in the Baseline Model

- **Continuous Control Variable**: `num_total_trains`
  - The total number of ingoing and outgoing traffic at a stop. We call this throughput later on.
- **Discrete Control Variable**: `day_of_week_name`
  - A categorical variable encoding the day of the week (e.g., Monday, Tuesday).

### Additional Control Variables in the Extended Models

- **Neighborhood Features**:
  - `total_distance_strength`, `total_time_strength`, `total_speed_strength` - The total distance, time, and speed of ingoing and outgoing edges for a node.
  - Strength is the graph terminology in weighted graphs for the sum of weights on the degrees - i.e., edges connected to the node.
  - `avg_neighboring_delay` - The average delay of incoming trains for neighboring stops (1 hop) around the node.
- **Christmas Dummy**: `christmas_period`
  - Categorical variable. True if the date is between the 22nd of December 2024 and the 2nd of January 2025.
