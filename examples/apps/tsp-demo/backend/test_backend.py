"""
Simple test script for TSP Demo backend
"""

import sys
import json
from app import HologramTSPSolver

def test_solver():
    """Test the TSP solver with a simple example"""
    print("Testing Hologram TSP Solver...")
    
    # Create simple test instance
    cities = [
        {'id': 0, 'name': 'City A', 'x': 0, 'y': 0},
        {'id': 1, 'name': 'City B', 'x': 1, 'y': 0},
        {'id': 2, 'name': 'City C', 'x': 1, 'y': 1},
        {'id': 3, 'name': 'City D', 'x': 0, 'y': 1},
    ]
    
    solver = HologramTSPSolver()
    try:
        solver.initialize()
        print(f"[OK] Solver initialized (Hologram version: {solver.executor is not None})")
        
        # Calculate distances
        distances = solver.calculate_distance_matrix(cities)
        print(f"[OK] Distance matrix calculated: {len(distances)}x{len(distances[0])}")
        
        # Solve TSP
        result = solver.solve_tsp(cities, distances, start_city=0)
        print(f"[OK] TSP solved successfully!")
        print(f"  - Tour: {result['cities']}")
        print(f"  - Distance: {result['distance']:.2f}")
        print(f"  - Runtime: {result['runtimeMs']:.2f}ms")
        print(f"  - Expression: {result['atlasExpression'][:50]}...")
        
        print("\n[SUCCESS] All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        solver.cleanup()

if __name__ == '__main__':
    success = test_solver()
    sys.exit(0 if success else 1)

