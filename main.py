import asyncio
from cProfile import Profile
from pstats import Stats
import time
import src.vessel_encounter as ve

SRC_PATH = './data/RealData1MRows.csv'
DISTANCE_THRESHOLD_IN_KM = 15
TEMPORAL_THRESHOLD_IN_SECONDS = 1
TIME_BETWEEN_EACH_DATA_SENT_IN_S = 1

def main():
    # Run the data stream
    asyncio.run(ve.vessel_encounters(SRC_PATH, DISTANCE_THRESHOLD_IN_KM, TEMPORAL_THRESHOLD_IN_SECONDS, TIME_BETWEEN_EACH_DATA_SENT_IN_S))

if __name__ == "__main__":
    with Profile() as pr:
        main()

    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(100, r"\((?!\_).*\)$")  # Exclude private and magic callables.
