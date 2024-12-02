import logging
import asyncio
import uvicorn
import argparse
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Vessel Server")

# FastAPI app
app = FastAPI(title="Vessel Clustering API with Background Fetching")

# Global variable to store current clusters
current_clusters: List[Dict[str, Any]] = []

# Configuration
EXTERNAL_ENDPOINT = "http://130.225.37.58:8000/dummy-ais-data"  # Replace with your actual external endpoint
FETCH_INTERVAL = 12  # Interval in seconds to fetch new data


async def fetch_and_process_data():
    """Fetch data from the external endpoint and process it to update `current_clusters`."""
    global current_clusters
    while True:
        try:
            # Fetch data from the external endpoint
            async with httpx.AsyncClient() as client:
                logger.info(f"Fetching data from {EXTERNAL_ENDPOINT}")
                response = await client.get(EXTERNAL_ENDPOINT)
                response.raise_for_status()
                data = response.json()
                print(data)
            
            # Process data to get current clusters
            logger.info("Processing fetched data...")
            #current_clusters = calculate_current_clusters(data)
            logger.info(f"Updated current clusters: {current_clusters}")
        except Exception as e:
            logger.error(f"Error fetching or processing data: {e}")
        
        await asyncio.sleep(FETCH_INTERVAL)


def calculate_current_clusters(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dummy function to calculate current clusters."""
    # Replace this with your actual clustering logic
    logger.info("Calculating current clusters...")
    return [{"id": i, "size": len(cluster)} for i, cluster in enumerate(data)]


@app.get("/clusters/current", tags=["Clustering"])
async def get_current_clusters():
    """Endpoint to get the current clusters."""
    global current_clusters
    if not current_clusters:
        return {"message": "No cluster data available yet. Please wait."}
    return {"clusters": current_clusters}


@asynccontextmanager
async def startup_event(app: FastAPI):
    """Start the background task to fetch and process data."""
    logger.info("Starting background task to fetch and process data.")
    asyncio.create_task(fetch_and_process_data())


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=4750)  


if __name__ == "__main__":

    run_server()
