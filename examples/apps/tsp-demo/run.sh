#!/bin/bash
# Run script for TSP Demo

echo "Starting TSP Demo..."

# Start backend in background
echo "Starting Python backend..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting React frontend..."
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT


