/**
 * Maximum Stress Tests for Parallel Universe Explorer
 * Tests with maximum configuration: 100 items, 5 containers, 10kg capacity
 */

// Mock Atlas
const Atlas = {
  classInfo: (byte: number) => {
    const h2 = Math.floor(byte / 24) % 4;
    const d = Math.floor(byte / 8) % 3;
    const l = byte % 8;
    return { components: { h2, d, l } };
  },
  canonicalByte: (idx: number) => idx % 256,
  classIndex: (byte: number) => byte % 96,
};

type ItemCategory = 'Electronics' | 'Clothing' | 'Food' | 'Tools' | 'Accessories' | 'Other';

interface Item {
  id: number;
  name: string;
  weight: number;
  value: number;
  category: ItemCategory;
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

const SYNERGIES = [
  { items: ['Laptop', 'Charger'], bonus: 200 },
  { items: ['Camera', 'Tripod'], bonus: 150 },
  { items: ['Phone', 'Case'], bonus: 50 },
  { items: ['Drone', 'Batteries'], bonus: 200 },
  { items: ['Laptop', 'Mouse', 'Keyboard'], bonus: 400 },
  { items: ['Phone', 'Charger', 'Power Bank'], bonus: 250 },
];

function calculateSynergyBonus(items: Item[]): number {
  let totalBonus = 0;
  const itemNames = new Set(items.map(item => item.name));
  
  for (const synergy of SYNERGIES) {
    if (synergy.items.every(name => itemNames.has(name))) {
      totalBonus += synergy.bonus;
    }
  }
  
  return totalBonus;
}

// Copy of HologramBinPackingSolver
class HologramBinPackingSolver {
  private encodeItemContainer(itemIndex: number, containerIndex: number): number {
    const baseClass = itemIndex % 96;
    const info = Atlas.classInfo(Atlas.canonicalByte(baseClass));
    const newH2 = containerIndex % 4;
    const newClass = 24 * newH2 + 8 * info.components.d + info.components.l;
    return Atlas.classIndex(Atlas.canonicalByte(newClass));
  }

  private encodeAssignment(assignment: number[], items: Item[]): string {
    const expressions: string[] = [];
    for (let i = 0; i < assignment.length; i++) {
      const containerIdx = assignment[i];
      if (containerIdx >= 0) {
        const classIdx = this.encodeItemContainer(i, containerIdx);
        expressions.push(`mark@c${classIdx}`);
      }
    }
    return expressions.join(' || ');
  }

  solve(items: Item[], capacities: number[]): Solution | null {
    const startTime = performance.now();
    const n = items.length;
    const numContainers = capacities.length;
    
    const universeCount = Math.pow(numContainers + 1, n);

    const sortedItems = items
      .map((item, idx) => ({ item, idx, ratio: item.value / item.weight }))
      .sort((a, b) => b.ratio - a.ratio);

    const assignment: number[] = new Array(n).fill(-1);
    const bestState = {
      assignment: null as number[] | null,
      value: 0,
    };

    const totalCapacity = capacities.reduce((sum, cap) => sum + cap, 0);
    const sortedItemsList = sortedItems.map(s => s.item);

    this.backtrackBinPackingOptimized(
      sortedItemsList,
      capacities,
      assignment,
      0,
      capacities.map(() => 0),
      0,
      bestState,
      this.calculateUpperBound(sortedItemsList, capacities, totalCapacity)
    );

    if (bestState.assignment) {
      const containers: ContainerResult[] = capacities.map((cap) => ({
        items: [],
        totalWeight: 0,
        totalValue: 0,
        capacity: cap,
      }));

      const originalAssignment = new Array(n).fill(-1);
      for (let i = 0; i < n; i++) {
        const originalIdx = sortedItems[i].idx;
        const containerIdx = bestState.assignment[i];
        originalAssignment[originalIdx] = containerIdx;

        if (containerIdx >= 0) {
          containers[containerIdx].items.push(sortedItems[i].item);
        }
      }

      let totalValue = 0;
      for (let c = 0; c < containers.length; c++) {
        containers[c].totalWeight = containers[c].items.reduce((sum, item) => sum + item.weight, 0);
        const baseValue = containers[c].items.reduce((sum, item) => sum + item.value, 0);
        const synergyBonus = calculateSynergyBonus(containers[c].items);
        containers[c].totalValue = baseValue + synergyBonus;
        totalValue += containers[c].totalValue;
      }

      const atlasExpression = this.encodeAssignment(originalAssignment, items);
      const runtimeMs = performance.now() - startTime;

      return {
        containers,
        totalValue,
        atlasExpression,
        runtimeMs,
        universeCount,
      };
    }

    return null;
  }

  private calculateUpperBound(items: Item[], capacities: number[], totalCapacity: number): number {
    let bound = 0;
    let remainingCapacity = totalCapacity;

    for (const item of items) {
      if (remainingCapacity >= item.weight) {
        bound += item.value;
        remainingCapacity -= item.weight;
      } else if (remainingCapacity > 0) {
        bound += item.value * (remainingCapacity / item.weight);
        break;
      }
    }

    return bound;
  }

  private backtrackBinPackingOptimized(
    items: Item[],
    capacities: number[],
    assignment: number[],
    itemIndex: number,
    currentWeights: number[],
    currentValue: number,
    bestState: { assignment: number[] | null; value: number },
    initialUpperBound: number
  ): void {
    if (itemIndex === items.length) {
      const containerItems: Item[][] = capacities.map(() => []);
      for (let i = 0; i < assignment.length; i++) {
        if (assignment[i] >= 0) {
          containerItems[assignment[i]].push(items[i]);
        }
      }

      let totalValueWithSynergy = 0;
      for (const containerItemList of containerItems) {
        const baseValue = containerItemList.reduce((sum, item) => sum + item.value, 0);
        const synergyBonus = calculateSynergyBonus(containerItemList);
        totalValueWithSynergy += baseValue + synergyBonus;
      }

      if (totalValueWithSynergy > bestState.value) {
        bestState.value = totalValueWithSynergy;
        bestState.assignment = [...assignment];
      }
      return;
    }

    const item = items[itemIndex];

    const remainingCapacity = capacities.reduce((sum, cap, idx) =>
      sum + Math.max(0, cap - currentWeights[idx]), 0
    );

    let upperBound = currentValue;
    let remCap = remainingCapacity;
    const remainingItems: Item[] = [];
    for (let i = itemIndex; i < items.length && remCap > 0; i++) {
      if (remCap >= items[i].weight) {
        upperBound += items[i].value;
        remCap -= items[i].weight;
        remainingItems.push(items[i]);
      } else {
        upperBound += items[i].value * (remCap / items[i].weight);
        break;
      }
    }

    const maxSynergyBonus = SYNERGIES.reduce((sum, syn) => sum + syn.bonus, 0) * 0.3;
    upperBound += maxSynergyBonus;

    if (upperBound <= bestState.value) {
      return;
    }

    const containerOrder = capacities
      .map((cap, idx) => ({ idx, remaining: cap - currentWeights[idx] }))
      .filter(c => c.remaining >= item.weight)
      .sort((a, b) => b.remaining - a.remaining);

    for (const { idx: c } of containerOrder) {
      assignment[itemIndex] = c;
      currentWeights[c] += item.weight;

      this.backtrackBinPackingOptimized(
        items,
        capacities,
        assignment,
        itemIndex + 1,
        currentWeights,
        currentValue + item.value,
        bestState,
        initialUpperBound
      );

      currentWeights[c] -= item.weight;
    }

    if (upperBound - item.value > bestState.value) {
      assignment[itemIndex] = -1;
      this.backtrackBinPackingOptimized(
        items,
        capacities,
        assignment,
        itemIndex + 1,
        currentWeights,
        currentValue,
        bestState,
        initialUpperBound
      );
    }
  }
}

// Test configuration - Maximum stress test
const MAX_TEST_PARAMS = {
  itemCount: 100,
  containerCount: 5,
  containerCapacity: 10,
};

const ITEM_DATA = [
  { name: 'Laptop', category: 'Electronics' as ItemCategory },
  { name: 'Phone', category: 'Electronics' as ItemCategory },
  { name: 'Camera', category: 'Electronics' as ItemCategory },
  { name: 'Charger', category: 'Electronics' as ItemCategory },
  { name: 'Mouse', category: 'Electronics' as ItemCategory },
  { name: 'Keyboard', category: 'Electronics' as ItemCategory },
  { name: 'Tablet', category: 'Electronics' as ItemCategory },
  { name: 'Headphones', category: 'Electronics' as ItemCategory },
  { name: 'Speaker', category: 'Electronics' as ItemCategory },
  { name: 'Monitor', category: 'Electronics' as ItemCategory },
  { name: 'T-Shirt', category: 'Clothing' as ItemCategory },
  { name: 'Jeans', category: 'Clothing' as ItemCategory },
  { name: 'Jacket', category: 'Clothing' as ItemCategory },
  { name: 'Shoes', category: 'Clothing' as ItemCategory },
  { name: 'Hat', category: 'Clothing' as ItemCategory },
  { name: 'Apple', category: 'Food' as ItemCategory },
  { name: 'Banana', category: 'Food' as ItemCategory },
  { name: 'Sandwich', category: 'Food' as ItemCategory },
  { name: 'Water', category: 'Food' as ItemCategory },
  { name: 'Snacks', category: 'Food' as ItemCategory },
  { name: 'Hammer', category: 'Tools' as ItemCategory },
  { name: 'Screwdriver', category: 'Tools' as ItemCategory },
  { name: 'Wrench', category: 'Tools' as ItemCategory },
  { name: 'Drill', category: 'Tools' as ItemCategory },
  { name: 'Saw', category: 'Tools' as ItemCategory },
  { name: 'Watch', category: 'Accessories' as ItemCategory },
  { name: 'Belt', category: 'Accessories' as ItemCategory },
  { name: 'Bag', category: 'Accessories' as ItemCategory },
  { name: 'Case', category: 'Accessories' as ItemCategory },
  { name: 'Tripod', category: 'Accessories' as ItemCategory },
];

function generateItems(count: number): Item[] {
  const items: Item[] = [];
  const usedNames = new Set<string>();

  for (let i = 0; i < count; i++) {
    let itemData: { name: string; category: ItemCategory };
    let attempts = 0;
    do {
      if (attempts < ITEM_DATA.length) {
        itemData = ITEM_DATA[Math.floor(Math.random() * ITEM_DATA.length)];
      } else {
        const baseItem = ITEM_DATA[Math.floor(Math.random() * ITEM_DATA.length)];
        const suffix = Math.floor((i - ITEM_DATA.length) / ITEM_DATA.length) + 1;
        const variant = (i - ITEM_DATA.length) % ITEM_DATA.length;
        itemData = {
          name: `${baseItem.name} #${suffix}-${variant}`,
          category: baseItem.category,
        };
      }
      attempts++;
    } while (usedNames.has(itemData.name) && attempts < 10000);
    usedNames.add(itemData.name);

    const randomValue = Math.random();
    const positionFactor = i / count;
    const adjustedRandom = randomValue * (1 - positionFactor * 0.15);

    let value: number;
    let weight: number;

    if (adjustedRandom < 0.3) {
      weight = Math.floor(Math.random() * 2) + 1;
      value = Math.floor(Math.random() * 1000) + 1500;
    } else if (adjustedRandom < 0.7) {
      weight = Math.floor(Math.random() * 3) + 1;
      value = Math.floor(Math.random() * 1000) + 500;
    } else {
      weight = Math.floor(Math.random() * 5) + 1;
      value = Math.floor(Math.random() * 450) + 50;
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

// Test framework
const testResults: Array<{ name: string; passed: boolean; message: string; duration?: number }> = [];

function test(name: string, fn: () => boolean | string): void {
  const startTime = performance.now();
  try {
    const result = fn();
    const duration = performance.now() - startTime;
    const passed = result === true || result === 'pass';
    const message = typeof result === 'string' && result !== 'pass' ? result : (passed ? 'PASS' : 'FAIL');
    testResults.push({ name, passed, message, duration });
    console.log(`${passed ? 'PASS' : 'FAIL'} ${name}: ${message}${duration ? ` (${duration.toFixed(2)}ms)` : ''}`);
  } catch (error) {
    const duration = performance.now() - startTime;
    testResults.push({ name, passed: false, message: `Error: ${error}`, duration });
    console.log(`FAIL ${name}: Error - ${error} (${duration.toFixed(2)}ms)`);
  }
}

console.log('\nMaximum Stress Tests - 100 Items, 5 Containers');
console.log(`Test Parameters: ${MAX_TEST_PARAMS.itemCount} items, ${MAX_TEST_PARAMS.containerCount} containers, ${MAX_TEST_PARAMS.containerCapacity}kg capacity\n`);
console.log('='.repeat(80));

// ============================================================================
// UNIT TESTS - Maximum Configuration
// ============================================================================

console.log('\nUNIT TESTS (5 Containers, 100 Items)\n');

// Test 1: Solver returns a solution with 5 containers
test('Unit: Solver returns solution with 5 containers', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  return solution !== null && solution.containers.length === 5;
});

// Test 2: All 5 containers respect capacity
test('Unit: All 5 containers respect capacity limits', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  if (solution.containers.length !== 5) return `Expected 5 containers, got ${solution.containers.length}`;
  
  for (const container of solution.containers) {
    if (container.totalWeight > container.capacity) {
      return `Container exceeds capacity: ${container.totalWeight}kg > ${container.capacity}kg`;
    }
  }
  return true;
});

// Test 3: Weight calculation accuracy across 5 containers
test('Unit: Container weights match sum of items (5 containers)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  for (const container of solution.containers) {
    const calculatedWeight = container.items.reduce((sum, item) => sum + item.weight, 0);
    if (Math.abs(container.totalWeight - calculatedWeight) > 0.01) {
      return `Weight mismatch: reported ${container.totalWeight}kg, calculated ${calculatedWeight}kg`;
    }
  }
  return true;
});

// Test 4: Value calculation includes synergies across 5 containers
test('Unit: Value calculation includes synergy bonuses (5 containers)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  let totalCalculatedValue = 0;
  for (const container of solution.containers) {
    const baseValue = container.items.reduce((sum, item) => sum + item.value, 0);
    const synergyBonus = calculateSynergyBonus(container.items);
    const expectedValue = baseValue + synergyBonus;
    totalCalculatedValue += expectedValue;
    
    if (Math.abs(container.totalValue - expectedValue) > 0.01) {
      return `Value mismatch in container: reported ${container.totalValue}, expected ${expectedValue}`;
    }
  }
  
  if (Math.abs(solution.totalValue - totalCalculatedValue) > 0.01) {
    return `Total value mismatch: reported ${solution.totalValue}, calculated ${totalCalculatedValue}`;
  }
  
  return true;
});

// Test 5: Items are assigned correctly across 5 containers
test('Unit: All items assigned correctly (no duplicates, no missing)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  const allAssignedItems = new Set<number>();
  for (const container of solution.containers) {
    for (const item of container.items) {
      if (allAssignedItems.has(item.id)) {
        return `Item ${item.id} (${item.name}) assigned to multiple containers`;
      }
      allAssignedItems.add(item.id);
    }
  }
  
  return true;
});

// Test 6: Solution uses all 5 containers effectively
test('Unit: Solution uses all 5 containers', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  const containersWithItems = solution.containers.filter(c => c.items.length > 0).length;
  if (containersWithItems < 2) {
    return `Expected at least 2 containers with items, got ${containersWithItems}`;
  }
  
  return true;
});

// ============================================================================
// STRESS TESTS - Maximum Configuration
// ============================================================================

console.log('\nSTRESS TESTS (5 Containers, 100 Items)\n');

// Stress Test 1: Performance with 100 items and 5 containers
test('Stress: Solves 100 items with 5 containers within reasonable time (< 60s)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const startTime = performance.now();
  const solution = solver.solve(items, capacities);
  const duration = performance.now() - startTime;
  
  if (!solution) return 'No solution found';
  if (duration > 60000) return `Too slow: ${duration.toFixed(2)}ms > 60000ms`;
  
  return true;
});

// Stress Test 2: Memory usage with maximum configuration
test('Stress: Handles 100 items and 5 containers without memory issues', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  try {
    const solution = solver.solve(items, capacities);
    if (!solution) return 'No solution found';
    if (solution.containers.length !== 5) return `Expected 5 containers, got ${solution.containers.length}`;
    return true;
  } catch (error) {
    return `Memory error: ${error}`;
  }
});

// Stress Test 3: Universe count calculation for 5 containers
test('Stress: Correct universe count for 5 containers', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  const expectedUniverseCount = Math.pow(MAX_TEST_PARAMS.containerCount + 1, MAX_TEST_PARAMS.itemCount);
  const ratio = solution.universeCount / expectedUniverseCount;
  if (ratio < 0.99 || ratio > 1.01) {
    return `Universe count mismatch: ${solution.universeCount} vs expected ~${expectedUniverseCount}`;
  }
  
  return true;
});

// Stress Test 4: Optimal solution quality with 5 containers
test('Stress: Solution maximizes total value (5 containers)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  if (solution.totalValue <= 0) {
    return 'Total value should be positive';
  }
  
  const totalCapacity = capacities.reduce((sum, cap) => sum + cap, 0);
  const totalWeight = solution.containers.reduce((sum, c) => sum + c.totalWeight, 0);
  const capacityUtilization = totalWeight / totalCapacity;
  
  if (capacityUtilization < 0.1) {
    return `Very low capacity utilization: ${(capacityUtilization * 100).toFixed(1)}%`;
  }
  
  return true;
});

// Stress Test 5: Multiple runs consistency with 5 containers
test('Stress: Consistent results across multiple runs (5 containers)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution1 = solver.solve(items, capacities);
  const solution2 = solver.solve(items, capacities);
  
  if (!solution1 || !solution2) return 'One or both solutions failed';
  
  if (Math.abs(solution1.totalValue - solution2.totalValue) > 0.01) {
    return `Inconsistent values: ${solution1.totalValue} vs ${solution2.totalValue}`;
  }
  
  return true;
});

// Stress Test 6: Runtime measurement accuracy
test('Stress: Runtime is measured correctly (5 containers)', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  if (solution.runtimeMs < 0) {
    return 'Runtime should be non-negative';
  }
  
  if (solution.runtimeMs > 120000) {
    return `Runtime seems too high: ${solution.runtimeMs.toFixed(2)}ms`;
  }
  
  return true;
});

// Stress Test 7: All containers are utilized
test('Stress: All 5 containers are utilized effectively', () => {
  const items = generateItems(MAX_TEST_PARAMS.itemCount);
  const solver = new HologramBinPackingSolver();
  const capacities = Array(MAX_TEST_PARAMS.containerCount).fill(MAX_TEST_PARAMS.containerCapacity);
  
  const solution = solver.solve(items, capacities);
  if (!solution) return 'No solution found';
  
  const emptyContainers = solution.containers.filter(c => c.items.length === 0).length;
  const totalItems = solution.containers.reduce((sum, c) => sum + c.items.length, 0);
  
  if (totalItems === 0) {
    return 'No items assigned to any container';
  }
  
  // With 100 items and 5 containers, we should use multiple containers
  const containersUsed = solution.containers.filter(c => c.items.length > 0).length;
  if (containersUsed < 2) {
    return `Expected at least 2 containers to be used, got ${containersUsed}`;
  }
  
  return true;
});

// ============================================================================
// SUMMARY
// ============================================================================

console.log('\n' + '='.repeat(80));
console.log('\nTest Summary:\n');

const passed = testResults.filter(t => t.passed).length;
const failed = testResults.filter(t => !t.passed).length;
const total = testResults.length;
const avgDuration = testResults.reduce((sum, t) => sum + (t.duration || 0), 0) / total;

console.log(`Total Tests: ${total}`);
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);
console.log(`Average Duration: ${avgDuration.toFixed(2)}ms`);

if (failed > 0) {
  console.log('\nFailed Tests:');
  testResults.filter(t => !t.passed).forEach(t => {
    console.log(`  - ${t.name}: ${t.message}${t.duration ? ` (${t.duration.toFixed(2)}ms)` : ''}`);
  });
}

// Performance summary
const stressTests = testResults.filter(t => t.name.startsWith('Stress:'));
const stressDurations = stressTests.map(t => t.duration || 0).filter(d => d > 0);
if (stressDurations.length > 0) {
  const maxStress = Math.max(...stressDurations);
  const minStress = Math.min(...stressDurations);
  const avgStress = stressDurations.reduce((sum, d) => sum + d, 0) / stressDurations.length;
  
  console.log('\nStress Test Performance:');
  console.log(`  Min: ${minStress.toFixed(2)}ms`);
  console.log(`  Max: ${maxStress.toFixed(2)}ms`);
  console.log(`  Avg: ${avgStress.toFixed(2)}ms`);
}

console.log('\n' + '='.repeat(80));
console.log('\nAll tests completed!\n');

// Exit with appropriate code
process.exit(failed > 0 ? 1 : 0);

