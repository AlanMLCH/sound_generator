# MusicGen (PyTorch + CUDA)

This project is an end-to-end MIDI music generator built with PyTorch. It includes a full pipeline for data processing, model training, and serving with a simple web UI.

For a detailed explanation of the project, please see the [full documentation](DOCUMENTATION.md).

## Features

- **End-to-End Pipeline**: From data ingestion to model training and evaluation.
- **Music Generation Model**: A PyTorch-based LSTM model for generating music sequences.
- **Web Interface**: A simple UI to generate music with custom instruments and parameters.
- **Dockerized**: The entire application is containerized for easy setup and deployment.

## Quickstart

### 1. Build Docker Containers

This command builds the Docker containers for the pipeline and services.

```bash
make docker-build
```

### 2. Run the Pipeline

Execute the following commands in order to download the data, preprocess it, and train the model.

```bash
# Download and prepare MIDI data
make ingest

# Preprocess the data and extract metadata
make preprocess

# Create features for the model
make fe

# Tune model hyperparameters
make tune

# Train the music generation model
make train

# Evaluate the trained model
make evaluate

# Run tests
make test
```

### 3. Start the Services

This command starts the backend API and the frontend web interface.

```bash
make up
```

Once the services are running, you can access the web interface at `http://localhost:8080`.

## Architecture

The project is divided into three main components:

1.  **Data Pipeline**: Handles the ingestion, preprocessing, and feature engineering of MIDI data.
2.  **Model Training**: Includes scripts for tuning, training, and evaluating the `MusicLSTM` model.
3.  **Serving**: Consists of a FastAPI backend to serve the model and a simple frontend for user interaction.

For a more detailed breakdown, see the [architecture section in the documentation](DOCUMENTATION.md#architecture).