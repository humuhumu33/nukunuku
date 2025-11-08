# Traveling Salesman Problem Demo - Hologram FFI Integration

Interactive web interface showcasing Hologram's geometric algebra approach for solving the Traveling Salesman Problem using the Hologram FFI Python backend.

## Architecture

- **Frontend**: React + TypeScript web application
- **Backend**: Python Flask API using Hologram FFI
- **Computation**: Hologram FFI for geometric algebra operations

## Setup

### 1. Install Frontend Dependencies

```bash
cd examples/apps/tsp-demo
npm install
```

### 2. Install Backend Dependencies

```bash
cd examples/apps/tsp-demo/backend
pip install -r requirements.txt
```

### 3. Build Hologram FFI (if not already built)

```bash
# From repo root
cargo build --workspace
```

## Running

### Start Backend (Terminal 1)

```bash
cd examples/apps/tsp-demo/backend
python app.py
```

The backend will start on `http://localhost:5000`

### Start Frontend (Terminal 2)

```bash
cd examples/apps/tsp-demo
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port if 5173 is busy)

## Testing

### Run Frontend Tests

```bash
cd examples/apps/tsp-demo
npm test
```

### Run Backend Tests

```bash
cd examples/apps/tsp-demo/backend
python -m pytest tests/ -v
```

## Features

- Interactive TSP solver using Hologram FFI
- Real-time visualization of optimal tours
- Performance comparison graphs
- Geometric encoding display
- Support for custom city sets and random distance generation

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/solve` - Solve TSP problem
- `POST /api/distances` - Calculate distance matrix

## License

Same as the main Hologram project.


