import asyncio
from cProfile import Profile
from pstats import Stats
import time
import encounters.vessel_encounter as ve
import pandas as pd

SRC_PATH = './data/RealData1MRows.csv'
DISTANCE_THRESHOLD_IN_KM = 15
TEMPORAL_THRESHOLD_IN_SECONDS = 4
TIME_FOR_BATCHES_IN_S = 2
RUN_UNTIL_TIMESTAMP = pd.to_datetime("2024-10-09 00:05:00")
RUN_FROM_TIMESTAMP = pd.to_datetime("2024-10-09 00:00:00")

def main():
    # Run the data stream
    asyncio.run(ve.vessel_encounters(SRC_PATH, DISTANCE_THRESHOLD_IN_KM, TEMPORAL_THRESHOLD_IN_SECONDS, TIME_FOR_BATCHES_IN_S, RUN_FROM_TIMESTAMP, RUN_UNTIL_TIMESTAMP))

if __name__ == "__main__":
    timestart = time.time()
    main()
    timeend = time.time()
    print("Time taken: ", timeend - timestart)
    quit()
    with Profile() as pr:
        main()

    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.