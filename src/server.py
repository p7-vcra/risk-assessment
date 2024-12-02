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
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# ENV variables
load_dotenv()
SOURCE_IP = os.getenv('SOURCE_IP', '0.0.0.0')
SOURCE_PORT = int(os.getenv('SOURCE_PORT', 4571))
LOG_LEVEL = os.getenv('SERVER_LOG_LEVEL', 'info')
SERVER_WORKERS = int(os.getenv('SERVER_WORKERS', 1))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Vessel Server")

# FastAPI app
app = FastAPI(title="Vessel Clustering API with Background Fetching")

# Global variable to store active pairs - pandas DataFrame
active_pairs = pd.DataFrame()

# Configuration
EXTERNAL_ENDPOINT = "http://130.225.37.58:8000/dummy-ais-data"  # Replace with your actual external endpoint
FETCH_INTERVAL = 1  # Interval in seconds to fetch new data
BATCH_PERIOD_IN_S = 12  # Collect data for 12 seconds before processing

# Global variable to track time and accumulated data
data_buffer = []
last_batch_time = None

async def fetch_and_process_data():
    """Fetch data from the external endpoint as a server-sent event and process it after accumulating for a set period."""
    global active_pairs, data_buffer, last_batch_time

    async with httpx.AsyncClient() as client:
        logger.info(f"Connecting to {EXTERNAL_ENDPOINT} for SSE stream...")

        try:
            async with client.stream("GET", EXTERNAL_ENDPOINT) as response:
                # Handle non-200 status codes early
                if response.status_code != 200:
                    logger.error(f"Error fetching data, status code {response.status_code}")
                    return

                async for line in response.aiter_lines():
                    if line.startswith('data:'):
                        await process_data_line(line)

        except httpx.RequestError as req_error:
            logger.error(f"Error with SSE stream connection: {req_error}")
        except Exception as e:
            logger.error(f"Unexpected error with SSE stream: {e}")

async def process_data_line(line):
    """Process the data line from the SSE stream and accumulate it for a set period."""
    global active_pairs, data_buffer, last_batch_time

    try:
        # Strip 'data:' prefix and parse the JSON data
        json_data = line[len('data:'):].strip()
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
        logger.error(f"Error processing data: {e}\n{traceback.format_exc()}")  # Log full stack trace

async def process_batch(batch_df):
    """Process the accumulated batch of data."""
    global active_pairs
    logger.info(f"Processing batch of data with {len(batch_df)} rows")

    active_pairs, output_pairs = ve.vessel_encounters_server(batch_df, active_pairs, temporal_threshold=1, distance_threshold=10)
    logger.info(f"Active pairs: {active_pairs}")
    logger.info(f"Output pairs: {output_pairs}")
    logger.info(f"Updated active pairs with {len(active_pairs)} encounters")

@app.get("/clusters/current", tags=["Clustering"])
async def get_clusters():
    """Endpoint to get the current clusters."""
    global active_pairs
    if active_pairs.empty:
        return {"message": "No cluster data available yet. Please wait."}
    return {"clusters": active_pairs.to_dict(orient="records")}

async def startup_event():
    """Start the background task to fetch and process data."""
    logger.info("Starting background task to fetch and process data.")
    asyncio.create_task(fetch_and_process_data())
    
app.add_event_handler("startup", startup_event)

def run_server():
    """Run the FastAPI server."""
    uvicorn.run(app, host=SOURCE_IP, port=SOURCE_PORT, log_level=LOG_LEVEL, workers=SERVER_WORKERS)

if __name__ == "__main__":
    run_server()
