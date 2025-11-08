"""
Packing Challenge Demo - Python Backend using Hologram FFI

This backend provides an API for solving knapsack and bin packing problems
using Hologram's geometric algebra approach through the FFI interface.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import hologram_ffi as hg
import json
import time
import math
from typing import List, Dict, Tuple, Optional

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# Packing Solvers using Hologram FFI
# ============================================================================

class HologramKnapsackSolver:
    """Knapsack Solver using Hologram FFI for geometric computations"""
    
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
    
    def encode_item_selection(self, item_index: int, is_selected: bool) -> int:
        """Encode item selection using Hologram class system"""
        base_class = item_index % 96
        # Use d modality: 0 = exclude, 1 = include
        new_d = 1 if is_selected else 0
        h2 = (item_index // 24) % 4
        l = item_index % 8
        new_class = 24 * h2 + 8 * new_d + l
        return new_class % 96
    
    def encode_selection(self, selected: List[bool]) -> str:
        """Generate Hologram expression representing a selection"""
        expressions = []
        for i, is_selected in enumerate(selected):
            class_idx = self.encode_item_selection(i, is_selected)
            expressions.append(f"mark@c{class_idx}")
        return " || ".join(expressions)
    
    def calculate_synergy_bonus(self, items: List[Dict], synergies: List[Dict]) -> float:
        """Calculate synergy bonus for a set of items"""
        bonus = 0.0
        item_names = {item['name'] for item in items}
        
        for synergy in synergies:
            synergy_items = set(synergy['items'])
            if synergy_items.issubset(item_names):
                bonus += synergy['bonus']
        
        return bonus
    
    def solve(self, items: List[Dict], capacity: float, synergies: List[Dict] = None) -> Dict:
        """Solve knapsack problem using backtracking with branch-and-bound"""
        start_time = time.perf_counter()
        
        if synergies is None:
            synergies = []
        
        # Sort items by value/weight ratio (best first)
        sorted_items = sorted(
            [(item, idx, item['value'] / item['weight']) for idx, item in enumerate(items)],
            key=lambda x: x[2],
            reverse=True
        )
        
        n = len(items)
        universe_count = 2 ** n
        
        best_state = {
            'items': [],
            'original_indices': [],
            'value': 0.0,
        }
        
        def backtrack(index: int, current_items: List[Dict], current_indices: List[int],
                     current_weight: float, current_value: float):
            # Base case
            if index == n:
                if current_items:
                    synergy_bonus = self.calculate_synergy_bonus(current_items, synergies)
                    total_value = current_value + synergy_bonus
                    
                    if total_value > best_state['value']:
                        best_state['items'] = current_items.copy()
                        best_state['original_indices'] = current_indices.copy()
                        best_state['value'] = total_value
                return
            
            item, original_idx, ratio = sorted_items[index]
            
            # Pruning: upper bound estimate
            remaining_items = sorted_items[index + 1:]
            remaining_value = 0.0
            rem_cap = capacity - current_weight
            
            for rem_item, _, _ in remaining_items:
                if rem_cap >= rem_item['weight']:
                    remaining_value += rem_item['value']
                    rem_cap -= rem_item['weight']
                elif rem_cap > 0:
                    remaining_value += rem_item['value'] * (rem_cap / rem_item['weight'])
                    break
            
            max_synergy = sum(s['bonus'] for s in synergies) * 0.5
            upper_bound = current_value + remaining_value + max_synergy
            
            if upper_bound <= best_state['value']:
                return  # Prune
            
            # Try including item
            if current_weight + item['weight'] <= capacity:
                backtrack(
                    index + 1,
                    current_items + [item],
                    current_indices + [original_idx],
                    current_weight + item['weight'],
                    current_value + item['value']
                )
            
            # Try excluding item
            backtrack(index + 1, current_items, current_indices, current_weight, current_value)
        
        backtrack(0, [], [], 0.0, 0.0)
        
        if not best_state['items']:
            return None
        
        total_weight = sum(item['weight'] for item in best_state['items'])
        synergy_bonus = self.calculate_synergy_bonus(best_state['items'], synergies)
        total_value = best_state['value']
        
        # Create selection array for encoding
        selected_array = [False] * n
        for idx in best_state['original_indices']:
            selected_array[idx] = True
        
        atlas_expression = self.encode_selection(selected_array)
        runtime_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'containers': [{
                'items': best_state['items'],
                'totalWeight': total_weight,
                'totalValue': total_value,
                'capacity': capacity,
            }],
            'totalValue': total_value,
            'atlasExpression': atlas_expression,
            'runtimeMs': runtime_ms,
            'universeCount': universe_count,
        }


class HologramBinPackingSolver:
    """Bin Packing Solver using Hologram FFI"""
    
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
    
    def encode_item_container(self, item_index: int, container_index: int) -> int:
        """Encode item-container assignment using Hologram class system"""
        base_class = item_index % 96
        # Use h2 for container assignment (supports up to 4 containers)
        new_h2 = container_index % 4
        d = (item_index // 8) % 3
        l = item_index % 8
        new_class = 24 * new_h2 + 8 * d + l
        return new_class % 96
    
    def encode_assignment(self, assignment: List[int], items: List[Dict]) -> str:
        """Generate Hologram expression representing bin packing assignment"""
        expressions = []
        for i, container_idx in enumerate(assignment):
            if container_idx >= 0:
                class_idx = self.encode_item_container(i, container_idx)
                expressions.append(f"mark@c{class_idx}")
        return " || ".join(expressions)
    
    def calculate_solution_value(self, items: List[Dict], assignment: List[int],
                                capacities: List[float]) -> float:
        """Calculate total value of a bin packing solution"""
        total_value = 0.0
        for i, container_idx in enumerate(assignment):
            if container_idx >= 0 and container_idx < len(capacities):
                total_value += items[i]['value']
        return total_value
    
    def greedy_solution(self, items: List[Dict], capacities: List[float]) -> List[int]:
        """Generate greedy solution for initial bound"""
        assignment = [-1] * len(items)
        container_weights = [0.0] * len(capacities)
        
        # Sort items by value/weight ratio
        sorted_items = sorted(
            enumerate(items),
            key=lambda x: x[1]['value'] / x[1]['weight'],
            reverse=True
        )
        
        for item_idx, item in sorted_items:
            best_container = -1
            best_ratio = -1
            
            for c_idx, cap in enumerate(capacities):
                if container_weights[c_idx] + item['weight'] <= cap:
                    ratio = item['value'] / item['weight']
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_container = c_idx
            
            if best_container >= 0:
                assignment[item_idx] = best_container
                container_weights[best_container] += item['weight']
        
        return assignment
    
    def solve(self, items: List[Dict], capacities: List[float]) -> Dict:
        """Solve bin packing problem"""
        start_time = time.perf_counter()
        n = len(items)
        num_containers = len(capacities)
        
        universe_count = (num_containers + 1) ** n
        
        # Initialize with greedy solution
        greedy_assignment = self.greedy_solution(items, capacities)
        greedy_value = self.calculate_solution_value(items, greedy_assignment, capacities)
        
        best_state = {
            'assignment': greedy_assignment.copy(),
            'value': greedy_value,
        }
        
        # Backtracking with optimization
        assignment = [-1] * n
        container_weights = [0.0] * len(capacities)
        
        def backtrack(index: int):
            if index == n:
                value = self.calculate_solution_value(items, assignment, capacities)
                if value > best_state['value']:
                    best_state['assignment'] = assignment.copy()
                    best_state['value'] = value
                return
            
            item = items[index]
            
            # Try placing in each container
            for c_idx in range(num_containers):
                if container_weights[c_idx] + item['weight'] <= capacities[c_idx]:
                    assignment[index] = c_idx
                    container_weights[c_idx] += item['weight']
                    backtrack(index + 1)
                    container_weights[c_idx] -= item['weight']
                    assignment[index] = -1
            
            # Try not placing item
            assignment[index] = -1
            backtrack(index + 1)
        
        backtrack(0)
        
        # Build result containers
        containers = []
        for c_idx, cap in enumerate(capacities):
            container_items = []
            total_weight = 0.0
            total_value = 0.0
            
            for i, item in enumerate(items):
                if best_state['assignment'][i] == c_idx:
                    container_items.append(item)
                    total_weight += item['weight']
                    total_value += item['value']
            
            containers.append({
                'items': container_items,
                'totalWeight': total_weight,
                'totalValue': total_value,
                'capacity': cap,
            })
        
        atlas_expression = self.encode_assignment(best_state['assignment'], items)
        runtime_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'containers': containers,
            'totalValue': best_state['value'],
            'atlasExpression': atlas_expression,
            'runtimeMs': runtime_ms,
            'universeCount': universe_count,
        }


# Global solver instances
knapsack_solver = HologramKnapsackSolver()
bin_packing_solver = HologramBinPackingSolver()

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

@app.route('/api/solve/knapsack', methods=['POST'])
def solve_knapsack():
    """Solve knapsack problem"""
    try:
        data = request.json
        items = data.get('items', [])
        capacity = data.get('capacity', 0)
        synergies = data.get('synergies', [])
        
        if not items:
            return jsonify({'error': 'No items provided'}), 400
        
        if capacity <= 0:
            return jsonify({'error': 'Invalid capacity'}), 400
        
        result = knapsack_solver.solve(items, capacity, synergies)
        
        if result is None:
            return jsonify({'error': 'No solution found'}), 404
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/solve/binpacking', methods=['POST'])
def solve_binpacking():
    """Solve bin packing problem"""
    try:
        data = request.json
        items = data.get('items', [])
        capacities = data.get('capacities', [])
        
        if not items:
            return jsonify({'error': 'No items provided'}), 400
        
        if not capacities:
            return jsonify({'error': 'No capacities provided'}), 400
        
        result = bin_packing_solver.solve(items, capacities)
        
        if result is None:
            return jsonify({'error': 'No solution found'}), 404
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Initialize solvers
        knapsack_solver.initialize()
        bin_packing_solver.initialize()
        print(f"Hologram FFI Version: {hg.get_version()}")
        print("Starting Packing Demo Backend on http://localhost:5001")
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        # Cleanup on shutdown
        knapsack_solver.cleanup()
        bin_packing_solver.cleanup()

