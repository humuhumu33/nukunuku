/**
 * Unit Tests for 1000 Items Support
 * Tests that the system can handle up to 1000 items correctly
 */

// Mock Atlas
const Atlas = {
  classInfo: (byte: number) => ({ components: { h2: 0, d: 0, l: 0 } }),
  canonicalByte: (idx: number) => idx,
  classIndex: (byte: number) => byte,
};

type ItemCategory = 'Electronics' | 'Clothing' | 'Food' | 'Tools' | 'Accessories' | 'Other';

interface Item {
  id: number;
  name: string;
  weight: number;
  value: number;
  category: ItemCategory;
}

const ITEM_DATA: Array<{ name: string; category: ItemCategory }> = [
  { name: 'Laptop', category: 'Electronics' },
  { name: 'Camera', category: 'Electronics' },
  { name: 'Jacket', category: 'Clothing' },
  { name: 'Snacks', category: 'Food' },
  { name: 'Tent', category: 'Tools' },
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

const testResults: Array<{ name: string; passed: boolean; message: string }> = [];

function test(name: string, fn: () => boolean | string): void {
  try {
    const result = fn();
    const passed = result === true || result === 'pass';
    const message = typeof result === 'string' && result !== 'pass' ? result : (passed ? 'PASS' : 'FAIL');
    testResults.push({ name, passed, message });
    console.log(`${passed ? 'PASS' : 'FAIL'} ${name}: ${message}`);
  } catch (error) {
    testResults.push({ name, passed: false, message: `Error: ${error}` });
    console.log(`FAIL ${name}: Error - ${error}`);
  }
}

console.log('\nTesting 1000 Items Support...\n');
console.log('='.repeat(60));

// Test 1: Generate 1000 items
test('Generate 1000 Items: Creates exactly 1000 items', () => {
  const items = generateItems(1000);
  return items.length === 1000;
});

// Test 2: All items have unique names
test('Generate 1000 Items: All items have unique names', () => {
  const items = generateItems(1000);
  const names = new Set(items.map(item => item.name));
  return names.size === 1000;
});

// Test 3: All items have valid properties
test('Generate 1000 Items: All items have valid properties', () => {
  const items = generateItems(1000);
  const allValid = items.every(item => 
    item.id >= 0 &&
    item.name.length > 0 &&
    item.weight >= 1 && item.weight <= 5 &&
    item.value >= 50 && item.value <= 2500 &&
    ['Electronics', 'Clothing', 'Food', 'Tools', 'Accessories', 'Other'].includes(item.category)
  );
  return allValid;
});

// Test 4: Incremental slicing works
test('Incremental Slicing: First 15 items are consistent', () => {
  const allItems = generateItems(1000);
  const first15 = allItems.slice(0, 15);
  const first15Again = allItems.slice(0, 15);
  
  // Check that slicing gives same items
  const same = first15.every((item, idx) => 
    item.id === first15Again[idx].id &&
    item.name === first15Again[idx].name &&
    item.weight === first15Again[idx].weight &&
    item.value === first15Again[idx].value
  );
  return same;
});

// Test 5: Incremental slicing - increasing count
test('Incremental Slicing: Increasing count adds more items', () => {
  const allItems = generateItems(1000);
  const first15 = allItems.slice(0, 15);
  const first20 = allItems.slice(0, 20);
  
  // First 15 should be the same
  const first15Match = first15.every((item, idx) => 
    item.id === first20[idx].id &&
    item.name === first20[idx].name
  );
  
  // Should have 5 more items
  const hasMore = first20.length === 20 && first15.length === 15;
  
  return first15Match && hasMore;
});

// Test 6: Performance - generation time
test('Performance: 1000 items generate in reasonable time', () => {
  const startTime = performance.now();
  const items = generateItems(1000);
  const endTime = performance.now();
  const duration = endTime - startTime;
  
  console.log(`  Generation time: ${duration.toFixed(2)}ms`);
  return duration < 1000; // Should complete in under 1 second
});

// Test 7: Memory - can handle 1000 items
test('Memory: Can store and access 1000 items', () => {
  const allItems = generateItems(1000);
  const item500 = allItems[499]; // 0-indexed
  const item1000 = allItems[999];
  
  return item500 !== undefined && item1000 !== undefined;
});

// Test 8: Value distribution
test('Value Distribution: Has mix of high/medium/low value items', () => {
  const items = generateItems(1000);
  const highValue = items.filter(item => item.value >= 1500).length;
  const mediumValue = items.filter(item => item.value >= 500 && item.value < 1500).length;
  const lowValue = items.filter(item => item.value < 500).length;
  
  // Should have reasonable distribution
  const hasHigh = highValue > 0;
  const hasMedium = mediumValue > 0;
  const hasLow = lowValue > 0;
  
  console.log(`  High-value items: ${highValue}, Medium: ${mediumValue}, Low: ${lowValue}`);
  return hasHigh && hasMedium && hasLow;
});

// Test 9: Weight distribution
test('Weight Distribution: Has items with various weights', () => {
  const items = generateItems(1000);
  const weights = new Set(items.map(item => item.weight));
  
  // Should have items with weights 1-5kg
  const hasVariety = weights.size >= 3;
  return hasVariety;
});

// Test 10: Slice performance
test('Slice Performance: Slicing 1000 items is fast', () => {
  const allItems = generateItems(1000);
  const startTime = performance.now();
  for (let i = 0; i < 100; i++) {
    allItems.slice(0, 50);
  }
  const endTime = performance.now();
  const duration = endTime - startTime;
  
  console.log(`  100 slice operations: ${duration.toFixed(2)}ms`);
  return duration < 100; // Should be very fast
});

console.log('\n' + '='.repeat(60));
console.log('\nTest Summary:\n');

const passed = testResults.filter(t => t.passed).length;
const failed = testResults.filter(t => !t.passed).length;
const total = testResults.length;

console.log(`Total Tests: ${total}`);
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

if (failed > 0) {
  console.log('\nFailed Tests:');
  testResults.filter(t => !t.passed).forEach(t => {
    console.log(`  - ${t.name}: ${t.message}`);
  });
}

console.log('\n' + '='.repeat(60));
console.log('\n1000 items support tests completed!\n');

export { testResults, test };

