# IBVAP — Intelligent Border Video Analytics Platform

An AI-powered border surveillance system prototype featuring an Edge AI inference pipeline (YOLOv8 + OpenCV), a FastAPI + PostgreSQL backend for alert ingestion, and a real-time Command & Control Dashboard.

---

## 🎯 Architecture Overview

`
┌────────────────────────────────────────────────────────┐
│  EDGE LAYER (edge_pipeline.py)                         │
│  • Webcam capture (OpenCV)                             │
│  • Smart frame skipping & YOLOv8 detection             │
│  • Sends Base64 JPEG + metadata to API on anomaly      │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST /api/events
┌──────────────────────────▼─────────────────────────────┐
│  BACKEND LAYER (main.py - FastAPI + PostgreSQL)        │
│  • Saves JPEGs to local filesystem (/snapshots)        │
│  • Stores event metadata & image paths in PostgreSQL   │
│  • Serves API endpoints & static dashboard assets      │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP GET /api/events (Polling)
┌──────────────────────────▼─────────────────────────────┐
│  C&C DASHBOARD (index.html + app.js)                   │
│  • Dark-mode military surveillance layout              │
│  • Real-time alert feed with bounding box overlays     │
│  • Threat level meter, stats, toast alerts & lightbox  │
└────────────────────────────────────────────────────────┘
`

---

## 🛠️ Prerequisites

Before installing, ensure you have the following installed on your system:

1. **Python 3.11+**: [Download Python](https://www.python.org/downloads/)
2. **Docker Desktop**: [Download Docker](https://www.docker.com/products/docker-desktop/) (required for running PostgreSQL 15)
3. **Webcam**: System camera for local video capture simulation

---

## 📥 Installation & Setup Instructions

Follow these step-by-step instructions to get IBVAP up and running locally.

### Step 1: Clone the Repository & Navigate to Workspace

`ash
git clone <repository-url>
cd prototype
`

### Step 2: Create and Activate a Virtual Environment (Recommended)

**On Windows (PowerShell):**
`powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
`

**On Linux / macOS:**
`ash
python3 -m venv venv
source venv/bin/activate
`

### Step 3: Install Dependencies

Install all required Python packages (FastAPI, OpenCV, Ultralytics YOLOv8, SQLAlchemy, asyncpg, etc.):

`ash
pip install -r requirements.txt
`

> **Note:** ultralytics>=8.3.0 is pinned to avoid compatibility issues with PyTorch 2.6+.

### Step 4: Environment Configuration

Create a .env file from the provided .env.example:

**On Windows (PowerShell):**
`powershell
Copy-Item .env.example .env
`

**On Linux / macOS:**
`ash
cp .env.example .env
`

Default configuration:
`env
DATABASE_URL=postgresql+asyncpg://ibvap_user:ibvap_secret@localhost:5432/ibvap_db
SNAPSHOTS_DIR=./snapshots
`

---

## 🚀 Running the System

To run the complete platform, open **three separate terminal windows**:

### Terminal 1: Launch PostgreSQL Database Container

Start the PostgreSQL 15 database using Docker Compose:

`ash
docker compose up -d
`

Verify that the database container is healthy:
`ash
docker ps
`
*(Look for ibvap_postgres with status Up (healthy))*

### Terminal 2: Start the FastAPI Backend Server

Run the FastAPI backend server with Uvicorn:

`ash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
`

Once running, access the following URLs in your web browser:
- **C&C Dashboard:** [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Terminal 3: Start the Edge AI Inference Pipeline

Run the edge pipeline script to start webcam video capture and object detection:

`ash
python edge_pipeline.py
`

* **First Run Note:** YOLOv8 model weights (yolov8n.pt, ~6.2 MB) will download automatically on first run.
* **Controls:** Press **q** in the OpenCV video feed window to stop the edge pipeline cleanly.

---

## 📂 Project Structure

`
prototype/
├── docker-compose.yml     # PostgreSQL 15 container definition
├── main.py                # FastAPI backend & database setup
├── edge_pipeline.py       # YOLOv8 webcam inference pipeline
├── index.html             # Command & Control Dashboard HTML layout
├── app.js                 # Dashboard frontend logic & polling engine
├── requirements.txt       # Python project dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
`

---

## 🛑 Stopping & Cleaning Up

### Stop the Edge Pipeline
Press **q** inside the OpenCV video window, or press Ctrl + C in Terminal 3.

### Stop the Backend Server
Press Ctrl + C in Terminal 2.

### Stop the PostgreSQL Container
In Terminal 1:
`ash
docker compose down
`

To remove stored database volumes as well:
`ash
docker compose down -v
`

---

## 📜 License

MIT License — Built for research and prototyping.
