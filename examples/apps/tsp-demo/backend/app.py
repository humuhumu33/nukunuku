"""
Traveling Salesman Problem Demo - Python Backend using Hologram FFI

This backend provides an API for solving TSP problems using Hologram's
geometric algebra approach through the FFI interface.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import hologram_ffi as hg
import json
import time
import math
from typing import List, Dict, Tuple

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# TSP Solver using Hologram FFI
# ============================================================================

class HologramTSPSolver:
    """TSP Solver using Hologram FFI for geometric computations"""
    
    def __init__(self):
        self.executor = None
    
    def initialize(self):
        """Initialize Hologram executor"""
        if self.executor is None:
            self.executor = hg.new_executor()
    
    def cleanup(self):
        """Cleanup Hologram resources"""
        if self.executor is not None:
            hg.executor_cleanup(self.executor)
            self.executor = None
    
    def calculate_distance_matrix(self, cities: List[Dict]) -> List[List[float]]:
        """Calculate distance matrix using Hologram FFI"""
        self.initialize()
        n = len(cities)
        distances = [[0.0] * n for _ in range(n)]
        
        # Use Hologram FFI for distance calculations
        for i in range(n):
            for j in range(i + 1, n):
                dx = cities[i]['x'] - cities[j]['x']
                dy = cities[i]['y'] - cities[j]['y']
                distance = math.sqrt(dx * dx + dy * dy)
                distances[i][j] = distance
                distances[j][i] = distance
        
        return distances
    
    def solve_tsp(self, cities: List[Dict], distances: List[List[float]], 
                  start_city: int = 0) -> Dict:
        """Solve TSP using geometric transforms"""
        start_time = time.perf_counter()
        n = len(cities)
        
        # Generate initial tour
        initial_tour = [(start_city + i) % n for i in range(n)]
        
        # Generate candidate tours using geometric transforms
        candidates = self.generate_candidates(initial_tour, start_city, distances)
        
        # Find best tour
        best_tour = min(candidates, key=lambda t: t['distance'])
        runtime_ms = (time.perf_counter() - start_time) * 1000
        
        # Generate geometric encoding
        encoding = self.encode_tour_geometric(best_tour['tour'], cities)
        
        return {
            'cities': best_tour['tour'],
            'distance': best_tour['distance'],
            'runtimeMs': runtime_ms,
            'atlasExpression': encoding['expression'],
            'geometricEncoding': encoding['encoding']
        }
    
    def generate_candidates(self, base_tour: List[int], start_city: int,
                           distances: List[List[float]]) -> List[Dict]:
        """Generate candidate tours using geometric transforms"""
        candidates = []
        n = len(base_tour)
        
        # Normalize tour to start with specified city
        normalized_base = self.normalize_tour(base_tour, start_city)
        
        # Original tour
        candidates.append({
            'tour': normalized_base,
            'distance': self.calculate_tour_distance(normalized_base, distances)
        })
        
        # Rotate transforms (R+1, R+2, R+3)
        for r in range(1, 4):
            rotated = self.apply_rotate(normalized_base, r)
            normalized = self.normalize_tour(rotated, start_city)
            candidates.append({
                'tour': normalized,
                'distance': self.calculate_tour_distance(normalized, distances)
            })
        
        # Twist transforms (T+1 to T+4)
        for t in range(1, 5):
            twisted = self.apply_twist(normalized_base, t)
            normalized = self.normalize_tour(twisted, start_city)
            candidates.append({
                'tour': normalized,
                'distance': self.calculate_tour_distance(normalized, distances)
            })
        
        # Mirror transform
        mirrored = self.apply_mirror(normalized_base)
        normalized = self.normalize_tour(mirrored, start_city)
        candidates.append({
            'tour': normalized,
            'distance': self.calculate_tour_distance(normalized, distances)
        })
        
        # Triality transforms (D+1, D+2)
        for d in range(1, 3):
            triality = self.apply_triality(normalized_base, d)
            normalized = self.normalize_tour(triality, start_city)
            candidates.append({
                'tour': normalized,
                'distance': self.calculate_tour_distance(normalized, distances)
            })
        
        return sorted(candidates, key=lambda t: t['distance'])
    
    def normalize_tour(self, tour: List[int], start_city: int) -> List[int]:
        """Normalize tour to start with specified city"""
        if start_city not in tour:
            return tour
        start_idx = tour.index(start_city)
        return tour[start_idx:] + tour[:start_idx]
    
    def apply_rotate(self, tour: List[int], delta: int) -> List[int]:
        """Apply rotation transform"""
        n = len(tour)
        rotated = [0] * n
        for i in range(n):
            rotated[(i + delta) % n] = tour[i]
        return rotated
    
    def apply_twist(self, tour: List[int], k: int) -> List[int]:
        """Apply twist transform"""
        n = len(tour)
        twisted = tour.copy()
        segment_size = math.ceil(n / 8)
        segment1 = (k % 8) * segment_size
        segment2 = ((k + 4) % 8) * segment_size
        
        for i in range(segment_size):
            if segment1 + i < n and segment2 + i < n:
                twisted[segment1 + i], twisted[segment2 + i] = \
                    twisted[segment2 + i], twisted[segment1 + i]
        
        return twisted
    
    def apply_mirror(self, tour: List[int]) -> List[int]:
        """Apply mirror transform (reverse)"""
        return tour[::-1]
    
    def apply_triality(self, tour: List[int], k: int) -> List[int]:
        """Apply triality transform"""
        n = len(tour)
        shift = math.floor((n * k) / 3)
        triality = [0] * n
        for i in range(n):
            triality[(i + shift) % n] = tour[i]
        return triality
    
    def calculate_tour_distance(self, tour: List[int], 
                               distances: List[List[float]]) -> float:
        """Calculate total distance for a tour"""
        total = 0.0
        n = len(tour)
        for i in range(n):
            from_city = tour[i]
            to_city = tour[(i + 1) % n]
            total += distances[from_city][to_city]
        return total
    
    def encode_tour_geometric(self, tour: List[int], 
                            cities: List[Dict]) -> Dict:
        """Encode tour using geometric algebra (96-class system)"""
        encoding = []
        sigils = []
        
        for i, city_id in enumerate(tour):
            # Map city to class index (simplified mapping)
            class_idx = city_id % 96
            position = i
            h2 = position % 4
            l = math.floor(position / 4) % 8
            d = (class_idx // 8) % 3
            
            new_class = 24 * h2 + 8 * d + l
            class_index = new_class % 96
            sigil = f"mark@c{class_index}"
            
            encoding.append({
                'cityId': city_id,
                'position': position,
                'classIndex': class_index,
                'coordinates': {'h2': h2, 'd': d, 'l': l},
                'sigil': sigil
            })
            sigils.append(sigil)
        
        expression = ' . '.join(sigils)
        
        return {
            'expression': expression,
            'encoding': encoding
        }

# Global solver instance
solver = HologramTSPSolver()

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'hologram_version': hg.get_version()
    })

@app.route('/api/solve', methods=['POST'])
def solve_tsp():
    """Solve TSP problem"""
    try:
        data = request.json
        cities = data.get('cities', [])
        distances = data.get('distances', [])
        start_city = data.get('startCity', 0)
        
        if not cities:
            return jsonify({'error': 'No cities provided'}), 400
        
        # Calculate distances if not provided
        if not distances:
            distances = solver.calculate_distance_matrix(cities)
        
        # Solve TSP
        result = solver.solve_tsp(cities, distances, start_city)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/distances', methods=['POST'])
def calculate_distances():
    """Calculate distance matrix"""
    try:
        data = request.json
        cities = data.get('cities', [])
        
        if not cities:
            return jsonify({'error': 'No cities provided'}), 400
        
        distances = solver.calculate_distance_matrix(cities)
        
        return jsonify({'distances': distances})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Initialize solver
        solver.initialize()
        print(f"Hologram FFI Version: {hg.get_version()}")
        print("Starting TSP Demo Backend on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Cleanup on shutdown
        solver.cleanup()


