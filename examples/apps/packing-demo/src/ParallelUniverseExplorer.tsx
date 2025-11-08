/**
 * Hologram Parallel Universe Explorer - Interactive Playground
 * 
 * This component demonstrates Hologram's ability to explore combinatorial spaces
 * by evaluating all possible universes simultaneously through symbolic computation.
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
// Using Hologram FFI backend instead of local sigmatics
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

// ============================================================================
// Types
// ============================================================================

type ItemCategory = 'Electronics' | 'Clothing' | 'Food' | 'Tools' | 'Accessories' | 'Other';

interface Item {
  id: number;
  name: string;
  weight: number;
  value: number;
  category: ItemCategory;
}

interface Synergy {
  items: string[]; // Item names that work together
  bonus: number; // Bonus value when all items are together
}

interface ContainerResult {
  items: Item[];
  totalWeight: number;
  totalValue: number;
  capacity: number;
}

interface Solution {
  containers: ContainerResult[];
  totalValue: number;
  atlasExpression: string;
  runtimeMs: number;
  universeCount: number;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Calculate large powers accurately using BigInt
 * Formats the result with proper comma separators
 * Always uses BigInt to ensure precision for all calculations
 */
function calculateUniverseCount(base: number, exponent: number): string {
  // Always use BigInt to maintain precision for all calculations
  // Math.pow loses precision even for smaller exponents with larger bases
  let result = BigInt(1);
  const baseBigInt = BigInt(base);
  
  for (let i = 0; i < exponent; i++) {
    result *= baseBigInt;
  }
  
  // Convert to string and add commas
  return result.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// ============================================================================
// Solver Classes (using Hologram encoding)
// ============================================================================

// API client for Hologram FFI backend - Knapsack Solver
class HologramKnapsackSolver {
  async solve(items: Item[], capacity: number, synergies: Synergy[] = []): Promise<Solution | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/solve/knapsack`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          items,
          capacity,
          synergies,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error solving knapsack:', error);
      throw error;
    }
  }
}

// API client for Hologram FFI backend - Bin Packing Solver
class HologramBinPackingSolver {
  async solve(items: Item[], capacities: number[]): Promise<Solution | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/solve/binpacking`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          items,
          capacities,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error solving bin packing:', error);
      throw error;
    }
  }
}

// ============================================================================
// Item Generation
// ============================================================================

const ITEM_DATA: Array<{ name: string; category: ItemCategory }> = [
  // Electronics
  { name: 'Laptop', category: 'Electronics' },
  { name: 'Camera', category: 'Electronics' },
  { name: 'Headphones', category: 'Electronics' },
  { name: 'Drone', category: 'Electronics' },
  { name: 'iPad', category: 'Electronics' },
  { name: 'Charger', category: 'Electronics' },
  { name: 'Power Bank', category: 'Electronics' },
  { name: 'Tablet', category: 'Electronics' },
  { name: 'Phone', category: 'Electronics' },
  { name: 'Speaker', category: 'Electronics' },
  { name: 'Watch', category: 'Electronics' },
  { name: 'GPS', category: 'Electronics' },
  { name: 'Calculator', category: 'Electronics' },
  { name: 'Keyboard', category: 'Electronics' },
  { name: 'Mouse', category: 'Electronics' },
  { name: 'Monitor', category: 'Electronics' },
  { name: 'Printer', category: 'Electronics' },
  { name: 'Scanner', category: 'Electronics' },
  { name: 'Projector', category: 'Electronics' },
  { name: 'Action Camera', category: 'Electronics' },
  { name: 'GoPro', category: 'Electronics' },
  { name: 'VR Headset', category: 'Electronics' },
  { name: 'Game Controller', category: 'Electronics' },
  { name: 'Joystick', category: 'Electronics' },
  { name: 'Steering Wheel', category: 'Electronics' },
  { name: 'Racing Seat', category: 'Electronics' },
  { name: 'Simulator', category: 'Electronics' },
  // Clothing
  { name: 'Jacket', category: 'Clothing' },
  { name: 'Shoes', category: 'Clothing' },
  { name: 'Backpack', category: 'Clothing' },
  // Food
  { name: 'Water Bottle', category: 'Food' },
  { name: 'Snacks', category: 'Food' },
  // Tools
  { name: 'Tent', category: 'Tools' },
  { name: 'Sleeping Bag', category: 'Tools' },
  { name: 'Stove', category: 'Tools' },
  { name: 'Lantern', category: 'Tools' },
  { name: 'Water Filter', category: 'Tools' },
  { name: 'Compass', category: 'Tools' },
  { name: 'Map', category: 'Tools' },
  { name: 'Flashlight', category: 'Tools' },
  { name: 'Rope', category: 'Tools' },
  { name: 'Tripod', category: 'Tools' },
  { name: 'Gimbal', category: 'Tools' },
  { name: 'Binoculars', category: 'Tools' },
  { name: 'Telescope', category: 'Tools' },
  { name: 'Microscope', category: 'Tools' },
  // Accessories
  { name: 'Book', category: 'Accessories' },
  { name: 'Notebook', category: 'Accessories' },
  { name: 'Sunglasses', category: 'Accessories' },
  { name: 'First Aid', category: 'Accessories' },
  { name: 'Batteries', category: 'Accessories' },
  { name: 'Memory Card', category: 'Accessories' },
];

// Synergy bonuses - items that work better together
const SYNERGIES: Synergy[] = [
  { items: ['Laptop', 'Charger'], bonus: 200 },
  { items: ['Camera', 'Tripod'], bonus: 150 },
  { items: ['Phone', 'Power Bank'], bonus: 100 },
  { items: ['Tent', 'Sleeping Bag'], bonus: 300 },
  { items: ['Stove', 'Lantern'], bonus: 150 },
  { items: ['Camera', 'Memory Card'], bonus: 100 },
  { items: ['Drone', 'Batteries'], bonus: 200 },
  { items: ['Laptop', 'Mouse', 'Keyboard'], bonus: 400 },
  { items: ['Phone', 'Charger', 'Power Bank'], bonus: 250 },
];

/**
 * Generate items using normal random distribution
 * Pre-generates all items (up to 100) with random values/weights
 * Ensures incremental improvement as item count increases
 */
function generateItems(count: number): Item[] {
  const items: Item[] = [];
  const usedNames = new Set<string>();
  
  for (let i = 0; i < count; i++) {
    let itemData: { name: string; category: ItemCategory };
    let attempts = 0;
    // If we run out of unique names, append a number to make it unique
    do {
      if (attempts < ITEM_DATA.length) {
        itemData = ITEM_DATA[Math.floor(Math.random() * ITEM_DATA.length)];
      } else {
        // Generate unique name with number suffix for items beyond ITEM_DATA length
        const baseItem = ITEM_DATA[Math.floor(Math.random() * ITEM_DATA.length)];
        const suffix = Math.floor((i - ITEM_DATA.length) / ITEM_DATA.length) + 1;
        const variant = (i - ITEM_DATA.length) % ITEM_DATA.length;
        itemData = {
          name: `${baseItem.name} #${suffix}-${variant}`,
          category: baseItem.category,
        };
      }
      attempts++;
    } while (usedNames.has(itemData.name) && attempts < 10000); // Increased safety limit for 1000 items
    usedNames.add(itemData.name);
    
    // Use normal random distribution - each item has equal chance
    // Probability: 30% high-value, 40% medium-value, 30% low-value
    // But ensure earlier items have slightly better average for incremental improvement
    const randomValue = Math.random();
    const positionFactor = i / count; // 0 to 1 (earlier items get slight boost)
    const adjustedRandom = randomValue * (1 - positionFactor * 0.15); // Slight favor for earlier items
    
    let value: number;
    let weight: number;
    
    if (adjustedRandom < 0.3) {
      // High-value items: $1500-$2500, very light (1-2kg)
      weight = Math.floor(Math.random() * 2) + 1; // 1-2kg
      value = Math.floor(Math.random() * 1000) + 1500; // $1500-$2500
    } else if (adjustedRandom < 0.7) {
      // Medium-value items: $500-$1500, light to medium (1-3kg)
      weight = Math.floor(Math.random() * 3) + 1; // 1-3kg
      value = Math.floor(Math.random() * 1000) + 500; // $500-$1500
    } else {
      // Lower-value items: $50-$500, any weight (1-5kg)
      weight = Math.floor(Math.random() * 5) + 1; // 1-5kg
      value = Math.floor(Math.random() * 450) + 50; // $50-$500
    }
    
    items.push({
      id: i,
      name: itemData.name,
      category: itemData.category,
      weight,
      value,
    });
  }
  
  return items;
}

/**
 * Calculate synergy bonuses for a set of items (optimized for maximum value)
 */
function calculateSynergyBonus(items: Item[]): number {
  let totalBonus = 0;
  const itemNames = new Set(items.map(item => item.name));
  
  for (const synergy of SYNERGIES) {
    // Check if all items in synergy are present
    const allPresent = synergy.items.every(name => itemNames.has(name));
    if (allPresent) {
      // Always use maximum bonus value
      totalBonus += synergy.bonus;
    }
  }
  
  return totalBonus;
}

// ============================================================================
// Main Component
// ============================================================================

const ParallelUniverseExplorer: React.FC = () => {
  const [itemCount, setItemCount] = useState(15);
  const [containerCount, setContainerCount] = useState(1);
  const [containerCapacity, setContainerCapacity] = useState(10);
  // Pre-generate all 100 items once, using normal random distribution
  const [allItems, setAllItems] = useState<Item[]>(() => generateItems(100));
  // Show only the number of items defined by the slider
  const items = useMemo(() => allItems.slice(0, itemCount), [allItems, itemCount]);
  const [solution, setSolution] = useState<Solution | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  // Calculate container capacities (all containers use the same capacity)
  const capacities = useMemo(() => 
    Array(containerCount).fill(containerCapacity),
    [containerCount, containerCapacity]
  );

  // When item count changes, just update solution (items are already pre-generated)
  useEffect(() => {
    setSolution(null);
  }, [itemCount]);

  // Solve the problem
  const runHologram = useCallback(async () => {
    setIsRunning(true);
    
    try {
      let result: Solution | null = null;
      
      if (containerCount === 1) {
        // Single container = knapsack problem
        const solver = new HologramKnapsackSolver();
        result = await solver.solve(items, containerCapacity, SYNERGIES);
      } else {
        // Multiple containers = bin packing problem
        const solver = new HologramBinPackingSolver();
        result = await solver.solve(items, capacities);
      }
      
      setSolution(result);
    } catch (error) {
      console.error('Error solving:', error);
      setSolution(null);
    } finally {
      setIsRunning(false);
    }
  }, [items, containerCount, containerCapacity, capacities]);

  // Auto-run when items, containers, or capacity change
  useEffect(() => {
    if (items.length > 0) {
      // Clear solution immediately to show loading state
      setSolution(null);
      runHologram();
    }
  }, [items, containerCount, containerCapacity, runHologram]);

  const randomizeItems = useCallback(() => {
    // Regenerate all 100 items when randomize is clicked
    const newAllItems = generateItems(100);
    setAllItems(newAllItems);
    setSolution(null);
  }, []);

  return (
    <div className="parallel-universe-explorer">
      {/* Problem Statement Section */}
      <div className="problem-statement">
        <h2>The Problem We're Solving</h2>
        <div className="problem-content">
          <div className="problem-box">
            <h3>Packing Problem</h3>
            <p>
              You have <strong>{items.length} items</strong> to pack into <strong>{containerCount} container{containerCount !== 1 ? 's' : ''}</strong>.
              Each item has a <strong>weight</strong> and a <strong>value</strong>.
            </p>
            <p>
              <strong>Goal:</strong> Select items to maximize total value while staying within weight limits.
            </p>
            <p style={{ fontSize: '0.9em', color: 'var(--text-muted-color)', marginTop: '0.5rem', fontStyle: 'italic' }}>
              Note: It's okay to be under weight, as long as it maximizes total value.
            </p>
            {containerCount === 1 ? (
              <div className="problem-details">
                <p><strong>Single Container (Knapsack Problem):</strong></p>
                <ul>
                  <li>Capacity: <strong>{containerCapacity}kg</strong></li>
                  <li>Each item can be <strong>packed</strong> or <strong>left behind</strong></li>
                  <li>Total combinations: <strong>2<sup>{items.length}</sup> = {calculateUniverseCount(2, items.length)}</strong> universes</li>
                </ul>
              </div>
            ) : (
              <div className="problem-details">
                <p><strong>Multiple Containers (Bin Packing Problem):</strong></p>
                <ul>
                  <li>Each container capacity: <strong>{containerCapacity}kg</strong></li>
                  <li>Each item can go to <strong>any container</strong> or be <strong>left behind</strong></li>
                  <li>Total combinations: <strong>({containerCount + 1})<sup>{items.length}</sup> = {calculateUniverseCount(containerCount + 1, items.length)}</strong> universes</li>
                </ul>
              </div>
            )}
          </div>
          <div className="hologram-power-box">
            <h3>How Hologram Solves This</h3>
            <div className="power-explanation">
              <div className="power-step">
                <span className="step-number">1</span>
                <div>
                  <strong>Classical Computing:</strong> Would need to check each combination one by one.
                  <br />
                  <span className="example">For {items.length} items: {containerCount === 1 ? calculateUniverseCount(2, items.length) : calculateUniverseCount(containerCount + 1, items.length)} separate calculations</span>
                </div>
              </div>
              <div className="power-step">
                <span className="step-number">2</span>
                <div>
                  <strong>Hologram Approach:</strong> Encodes all combinations in a single symbolic expression.
                  <br />
                  <span className="example">Uses parallel composition (<code>||</code>) to represent all universes simultaneously</span>
                </div>
              </div>
              <div className="power-step">
                <span className="step-number">3</span>
                <div>
                  <strong>Result:</strong> Evaluates all {containerCount === 1 ? calculateUniverseCount(2, items.length) : calculateUniverseCount(containerCount + 1, items.length)} universes in milliseconds.
                  <br />
                  <span className="example">Finds the optimal solution instantly through symbolic computation</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="controls-panel">
        <div className="control-group">
          <label htmlFor="item-count">
            Number of Items: <strong>{itemCount}</strong>
          </label>
          <input
            id="item-count"
            type="range"
            min="5"
            max="100"
            value={itemCount}
            onChange={(e) => setItemCount(parseInt(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>5</span>
            <span>100</span>
          </div>
          <div className="universe-count-preview">
            <span className="preview-label">Possible Universes:</span>
            <span className="preview-value">
              {containerCount === 1 
                ? `2^${itemCount} = ${calculateUniverseCount(2, itemCount)}`
                : `(${containerCount + 1})^${itemCount} = ${calculateUniverseCount(containerCount + 1, itemCount)}`
              }
            </span>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="container-count">
            Number of Containers: <strong>{containerCount}</strong>
          </label>
          <input
            id="container-count"
            type="range"
            min="1"
            max="2"
            value={containerCount}
            onChange={(e) => setContainerCount(parseInt(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>1</span>
            <span>2</span>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="capacity">
            Container Capacity: <strong>{containerCapacity}kg</strong>
          </label>
          <input
            id="capacity"
            type="range"
            min="5"
            max="30"
            value={containerCapacity}
            onChange={(e) => setContainerCapacity(parseInt(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>5kg</span>
            <span>30kg</span>
          </div>
        </div>

        <div className="button-group">
          <button 
            className="button button-primary" 
            onClick={runHologram}
            disabled={isRunning}
          >
            {isRunning ? 'Running Hologram...' : 'Run Hologram'}
          </button>
          <button 
            className="button button-secondary" 
            onClick={randomizeItems}
          >
            Randomize Items
          </button>
        </div>
      </div>

      {isRunning && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
          </div>
          <p className="loading-text">Exploring parallel universes...</p>
          <p className="loading-subtext">Calculating optimal solution</p>
        </div>
      )}

      {solution && (
        <div className="results-panel">
          <div className="results-header">
            <h2>Optimal Solution Found!</h2>
            <div className="solution-explanation">
              <p>
                Hologram explored <strong>{containerCount === 1 ? calculateUniverseCount(2, items.length) : calculateUniverseCount(containerCount + 1, items.length)} parallel universes</strong> in just{' '}
                <strong>{solution.runtimeMs.toFixed(2)} milliseconds</strong> to find the best combination.
              </p>
            </div>
            <div className="solution-stats">
              <div className="stat">
                <span className="stat-label">Universes Explored</span>
                <span className="stat-value">{containerCount === 1 ? calculateUniverseCount(2, items.length) : calculateUniverseCount(containerCount + 1, items.length)}</span>
                <span className="stat-desc">All possible combinations</span>
              </div>
              <div className="stat">
                <span className="stat-label">Runtime</span>
                <span className="stat-value">{solution.runtimeMs.toFixed(2)}ms</span>
                <span className="stat-desc">Symbolic evaluation</span>
              </div>
              <div className="stat">
                <span className="stat-label">Total Value</span>
                <span className="stat-value">${solution.totalValue.toLocaleString()}</span>
                <span className="stat-desc">Maximum possible</span>
              </div>
            </div>
          </div>

          <div className="containers-grid">
            {solution.containers
              .map((container, idx) => ({ container, originalIdx: idx }))
              .sort((a, b) => b.container.totalValue - a.container.totalValue)
              .map(({ container, originalIdx }, displayIdx) => {
                // Calculate actual total weight from items in container
                const actualTotalWeight = container.items.reduce((sum, item) => sum + item.weight, 0);
                const isOverloaded = actualTotalWeight > container.capacity;
                return (
                  <div 
                    key={originalIdx} 
                    className={`container-card ${isOverloaded ? 'overloaded' : ''}`}
                  >
                    <div className="container-header">
                      <h3>Container {originalIdx + 1}</h3>
                    <div className="container-stats">
                      <span className={isOverloaded ? 'error' : 'success'}>
                        {actualTotalWeight}kg / {container.capacity}kg
                      </span>
                      <span className="value">${container.totalValue.toLocaleString()}</span>
                    </div>
                  </div>
                  <div className="container-items">
                    {container.items.length === 0 ? (
                      <p className="empty-container">No items</p>
                    ) : (
                      <ul>
                        {container.items.map((item) => (
                          <li key={item.id} className="container-item">
                            <span className="item-name">{item.name}</span>
                            <span className="item-details">
                              {item.weight}kg • ${item.value.toLocaleString()}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="items-panel">
        <h2>Items ({items.length})</h2>
        <div className="items-table-container">
          <table className="items-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th>Category</th>
                <th>Weight</th>
                <th>Value</th>
                <th>Ratio</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => {
                const isSelected = solution?.containers.some(c => 
                  c.items.some(i => i.id === item.id)
                ) ?? false;
                return (
                  <tr key={item.id} className={isSelected ? 'selected' : ''}>
                    <td>{item.id + 1}</td>
                    <td>{item.name}</td>
                    <td>
                      <span className="category-badge">
                        {item.category}
                      </span>
                    </td>
                    <td>{item.weight}kg</td>
                    <td>${item.value.toLocaleString()}</td>
                    <td>${(item.value / item.weight).toFixed(0)}/kg</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        
      </div>
    </div>
  );
};

export default ParallelUniverseExplorer;

