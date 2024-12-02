# Risk-assessment

This repo supports risk-assessment in the larger context of VCRA P7 Project.

## Features

- [x] Temporal + Spatial Clustering
- [x] Model to calculate VCRA
- [ ] Server to provide VCRA + PVCRA

## Project Structure

```
. 
├── .env.example         # Example environment file 
├── .env                 # environment file 
├── requirements.txt     # Python dependencies 
├── src/                 # Main project source files 
│   ├── utils/           # Utility functions and scripts 
│   ├── vcra/            # Core VCRA calculation logic 
│   ├── helper_scripts/  # Scripts for data preprocessing 
│   ├── encounters/      # Vessel encounter detection logic 
├── tests/               # Unit tests 
│   ├── encounters/      # Tests specific to encounters module 
├── data/                # Folder to store raw or intermediate data - defined in .env
└── output/              # Folder for processed output files - defined in .env
```

## Environment and Dependencies

- The repository includes an example environment file (`.env.example`) that should be copied to `.env` and configured with project-specific values.
- Dependencies are listed in `requirements.txt` and can be installed using pip:

    ```bash
    pip install -r requirements.txt
    ```

### Example Usage

Run the main processing:

```bash
python main.py
```

Run with profiling enabled:

```bash
python main.py --profile
```

Create training data:

```bash
python main.py --create-training-data
```

## Data Format

The input AIS data should be supplied in the following CSV format :

```
Timestamp,MMSI,Latitude,Longitude,SOG,COG,Length
```

## Testing

Tests are written using pytest. Run the tests using the following command:

```bash
pytest
```

## Logging

Logging is configured using a custom logger (`setup_logger`) and logs are written to the console. Modify `utils/logger.py` to adjust log settings.

## Profiling

The script supports profiling to analyze performance. Profiling results are printed to the console with the top 100 cumulative time functions.
