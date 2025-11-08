# Packing Challenge Demo

This demo showcases Hologram FFI's ability to solve knapsack and bin packing problems using geometric algebra.

## Features

- **Knapsack Problem**: Single container optimization
- **Bin Packing Problem**: Multiple container optimization
- **Synergy Bonuses**: Items that work better together
- **Real-time Visualization**: See solutions as they're computed

## Running the Demo

### Prerequisites

- Python 3.8+ with `hologram_ffi` installed
- Node.js 18+ and npm

### Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
npm install
```

### Running

**Windows:**
```bash
run.bat
```

**Linux/macOS:**
```bash
./run.sh
```

Or manually:

1. Start the backend (port 5001):
```bash
cd backend
python app.py
```

2. Start the frontend (port 3001):
```bash
npm run dev
```

3. Open http://localhost:3001 in your browser

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/solve/knapsack` - Solve knapsack problem
- `POST /api/solve/binpacking` - Solve bin packing problem

