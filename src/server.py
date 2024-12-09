import logging
import asyncio
import uvicorn
import httpx
import json
import pandas as pd
import time
import traceback
import os
import encounters.vessel_encounter as ve
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# ENV variables
load_dotenv()
TARGET_ENDPOINT_FOR_CURRENT_SHIPS = os.getenv("TARGET_ENDPOINT_FOR_CURRENT_SHIPS")
DISTANCE_THRESHOLD_IN_KM = float(os.getenv("DISTANCE_THRESHOLD_IN_KM", 1))
TEMPOERAL_THRESHOLD_IN_SECONDS = float(os.getenv("TEMPOERAL_THRESHOLD_IN_SECONDS", 30))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Vessel Server")

# FastAPI app
app = FastAPI(title="Vessel Clustering API with Background Fetching")

# Global variable to store active pairs - pandas DataFrame
active_pairs: pd.DataFrame = pd.DataFrame()
active_pairs_future: pd.DataFrame = pd.DataFrame()
output_pairs_current: pd.DataFrame = pd.DataFrame()
output_pairs_future: pd.DataFrame = pd.DataFrame()

# Configuration
EXTERNAL_ENDPOINT = (
    TARGET_ENDPOINT_FOR_CURRENT_SHIPS  # Replace with your actual external endpoint
)
FETCH_INTERVAL = 1  # Interval in seconds to fetch new data
BATCH_PERIOD_IN_S = 30  # Collect data for x seconds before processing

# Global variable to track time and accumulated data
data_buffer = []
last_batch_time = None


async def fetch_and_process_data():
    """Fetch data from the external endpoint as a server-sent event and process it after accumulating for a set period."""
    global active_pairs, data_buffer, last_batch_time

    while True:  # Retry loop
        async with httpx.AsyncClient() as client:
            logger.info(f"Connecting to {EXTERNAL_ENDPOINT} for SSE stream...")

            try:
                async with client.stream("GET", EXTERNAL_ENDPOINT) as response:
                    # Handle non-200 status codes early
                    if response.status_code != 200:
                        logger.error(
                            f"Error fetching data, status code {response.status_code}"
                        )
                        raise httpx.RequestError(
                            f"Invalid status code: {response.status_code}"
                        )

                    event = None
                    async for line in response.aiter_lines():
                        # save the event type
                        if line.startswith("event:"):
                            event = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            if event == "ais":
                                await process_data_line(line)
                                event = None

            except httpx.RequestError as req_error:
                logger.error(f"Error with SSE stream connection: {req_error}")
            except Exception as e:
                logger.error(f"Unexpected error with SSE stream: {e}")

            # Clear cache and reset global variables
            logger.warning("Clearing cache and resetting due to connection failure.")
            active_pairs = pd.DataFrame()
            data_buffer = []
            last_batch_time = None

            # Delay before retrying to avoid rapid reconnect attempts
            logger.info("Retrying connection in 5 seconds...")
            await asyncio.sleep(5)


async def process_data_line(line):
    """Process the data line from the SSE stream and accumulate it for a set period."""
    global active_pairs, data_buffer, last_batch_time

    try:
        # Strip 'event: ais\n data:' from the line
        json_data = line[len("data: ") :].strip()
        data = json.loads(json_data)

        logger.info(f"Received data: {len(data)}")

        # Convert the data to a pandas DataFrame if it's in a suitable format
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list) and all(isinstance(i, dict) for i in data):
            df = pd.DataFrame(data)
        else:
            logger.warning(f"Unexpected data format: {data}")
            return

        # Add the data to the buffer
        data_buffer.append(df)

        # Track the time when the first piece of data is received
        if last_batch_time is None:
            last_batch_time = time.time()

        # Check if we have accumulated enough data (BATCH_PERIOD_IN_S)
        if time.time() - last_batch_time >= BATCH_PERIOD_IN_S:
            # Concatenate all the buffered data into one DataFrame
            batch_df = pd.concat(data_buffer, ignore_index=True)

            # Process the batch data
            await process_batch(batch_df)

            # Reset the buffer and time tracker
            data_buffer = []
            last_batch_time = None

    except json.JSONDecodeError as json_error:
        logger.error(f"Error decoding JSON: {json_error}")
    except Exception as e:
        logger.error(
            f"Error processing data: {e}\n{traceback.format_exc()}"
        )  # Log full stack trace


async def process_batch(batch_df):
    """Process the accumulated batch of data."""
    global active_pairs, output_pairs_current
    logger.info(f"Processing batch of data with {len(batch_df)} rows")

    active_pairs, output_pairs_current = ve.vessel_encounters_server(
        batch_df,
        active_pairs,
        temporal_threshold=TEMPOERAL_THRESHOLD_IN_SECONDS,
        distance_threshold=DISTANCE_THRESHOLD_IN_KM,
    )
    if not output_pairs_current.empty:
        logger.info(f"Ouput pairs: {output_pairs_current}")


@app.get("/clusters/current", tags=["Clustering"])
async def get_clusters():
    """Endpoint to get the current clusters."""
    global output_pairs_current
    if output_pairs_current.empty:
        # Return empty JSON if no clusters are found
        return {"clusters": []}

    output_pairs_current = output_pairs_current.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Convert the DataFrame to JSON-compatible list of dicts, including the index
    clusters = {
        "clusters": output_pairs_current.reset_index().to_dict(orient="records")
    }
    return clusters


@app.get("/clusters/future", tags=["Clustering"])
async def get_future_clusters():
    """Endpoint to get the future clusters."""
    global output_pairs_future
    if output_pairs_future.empty:
        # Return empty JSON if no clusters are found
        return {"clusters": []}

    output_pairs_future = output_pairs_future.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Convert the DataFrame to JSON-compatible list of dicts, including the index
    clusters = {"clusters": output_pairs_future.reset_index().to_dict(orient="records")}
    return clusters


async def startup_event():
    """Start the background task to fetch and process data."""
    logger.info("Starting background task to fetch and process data.")
    asyncio.create_task(fetch_and_process_data())


app.add_event_handler("startup", startup_event)
