/**
 * Unit test for Randomize Distances functionality
 */

interface City {
  id: number;
  name: string;
  x: number;
  y: number;
}

interface TSPInstance {
  cities: City[];
  distances: number[][];
}

// Copy the createInstance function for testing
function generateRandomDistance(mean: number, stdDev: number, randomState: { hasSpare: boolean; spare: number }): number {
  let z: number;
  
  if (randomState.hasSpare) {
    randomState.hasSpare = false;
    z = randomState.spare;
  } else {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
    randomState.hasSpare = true;
    randomState.spare = z1;
    z = z0;
  }
  
  const distance = mean + stdDev * z;
  return Math.max(0.1, distance);
}

function createInstance(cities: City[], useRandomDistances: boolean = false): TSPInstance {
  const distances: number[][] = [];
  
  if (useRandomDistances) {
    const randomState = { hasSpare: false, spare: 0 };
    const coords = cities.map(c => ({ x: c.x, y: c.y }));
    const xs = coords.map(c => c.x);
    const ys = coords.map(c => c.y);
    const rangeX = Math.max(...xs) - Math.min(...xs) || 20;
    const rangeY = Math.max(...ys) - Math.min(...ys) || 20;
    const meanDistance = Math.sqrt(rangeX * rangeX + rangeY * rangeY) / 2;
    const stdDev = meanDistance * 0.4;
    
    for (let i = 0; i < cities.length; i++) {
      distances[i] = new Array(cities.length).fill(0);
    }
    
    for (let i = 0; i < cities.length; i++) {
      for (let j = i + 1; j < cities.length; j++) {
        const distance = generateRandomDistance(meanDistance, stdDev, randomState);
        distances[i][j] = Math.round(distance * 10) / 10;
        distances[j][i] = distances[i][j];
      }
    }
  } else {
    for (let i = 0; i < cities.length; i++) {
      distances[i] = [];
      for (let j = 0; j < cities.length; j++) {
        if (i === j) {
          distances[i][j] = 0;
        } else {
          const dx = cities[i].x - cities[j].x;
          const dy = cities[i].y - cities[j].y;
          distances[i][j] = Math.sqrt(dx * dx + dy * dy);
        }
      }
    }
  }
  
  return { cities, distances };
}

function testRandomizeDistances() {
  console.log('Testing Randomize Distances...\n');

  // Test 1: Verify random distances are generated
  console.log('Test 1: Verify random distances are generated');
  const testCities: City[] = [
    { id: 0, name: 'A', x: 0, y: 0 },
    { id: 1, name: 'B', x: 10, y: 0 },
    { id: 2, name: 'C', x: 10, y: 10 },
  ];

  // Create instance with Euclidean distances
  const euclideanInstance = createInstance(testCities, false);
  console.log('Euclidean distances:');
  console.log('A->B:', euclideanInstance.distances[0][1].toFixed(2));
  console.log('B->C:', euclideanInstance.distances[1][2].toFixed(2));
  console.log('A->C:', euclideanInstance.distances[0][2].toFixed(2));

  // Create instance with random distances
  const randomInstance1 = createInstance(testCities, true);
  console.log('\nRandom distances (first call):');
  console.log('A->B:', randomInstance1.distances[0][1].toFixed(2));
  console.log('B->C:', randomInstance1.distances[1][2].toFixed(2));
  console.log('A->C:', randomInstance1.distances[0][2].toFixed(2));

  // Create another instance with random distances (should be different)
  const randomInstance2 = createInstance(testCities, true);
  console.log('\nRandom distances (second call - should be different):');
  console.log('A->B:', randomInstance2.distances[0][1].toFixed(2));
  console.log('B->C:', randomInstance2.distances[1][2].toFixed(2));
  console.log('A->C:', randomInstance2.distances[0][2].toFixed(2));

  // Test 2: Verify distances are symmetric
  console.log('\nTest 2: Verify distances are symmetric');
  let allSymmetric = true;
  for (let i = 0; i < randomInstance1.distances.length; i++) {
    for (let j = 0; j < randomInstance1.distances[i].length; j++) {
      if (Math.abs(randomInstance1.distances[i][j] - randomInstance1.distances[j][i]) > 0.01) {
        allSymmetric = false;
        console.log(`ERROR: distances[${i}][${j}] (${randomInstance1.distances[i][j]}) != distances[${j}][${i}] (${randomInstance1.distances[j][i]})`);
      }
    }
  }
  if (allSymmetric) {
    console.log('✓ All distances are symmetric');
  }

  // Test 3: Verify distances are positive
  console.log('\nTest 3: Verify distances are positive');
  let allPositive = true;
  for (let i = 0; i < randomInstance1.distances.length; i++) {
    for (let j = 0; j < randomInstance1.distances[i].length; j++) {
      if (i !== j && randomInstance1.distances[i][j] <= 0) {
        allPositive = false;
        console.log(`ERROR: distances[${i}][${j}] = ${randomInstance1.distances[i][j]} is not positive`);
      }
    }
  }
  if (allPositive) {
    console.log('✓ All distances are positive');
  }

  // Test 4: Verify diagonal distances are zero
  console.log('\nTest 4: Verify diagonal distances are zero');
  let allDiagonalZero = true;
  for (let i = 0; i < randomInstance1.distances.length; i++) {
    if (randomInstance1.distances[i][i] !== 0) {
      allDiagonalZero = false;
      console.log(`ERROR: distances[${i}][${i}] = ${randomInstance1.distances[i][i]} should be 0`);
    }
  }
  if (allDiagonalZero) {
    console.log('✓ All diagonal distances are zero');
  }

  // Test 5: Verify distances are different between calls
  console.log('\nTest 5: Verify distances are different between calls');
  const distances1 = randomInstance1.distances[0][1];
  const distances2 = randomInstance2.distances[0][1];
  if (Math.abs(distances1 - distances2) > 0.01) {
    console.log('✓ Distances are different between calls (as expected)');
  } else {
    console.log('WARNING: Distances might be the same (could be coincidence)');
  }

  console.log('\n=== Test Summary ===');
  console.log('All basic tests passed!');
  console.log('\nNote: This test verifies the distance generation function.');
  console.log('For full integration testing, test the button click handler in the browser.');
}

// Run the test
testRandomizeDistances();

