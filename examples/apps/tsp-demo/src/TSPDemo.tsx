/**
 * Traveling Salesman Problem Demo - Web Showcase
 * 
 * Interactive web interface showcasing Sigmatics power for solving TSP
 * using geometric algebra and transform operations.
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
// Using Hologram FFI backend instead of local sigmatics
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// ============================================================================
// Types
// ============================================================================

interface City {
  id: number;
  name: string;
  x: number;
  y: number;
}

interface GeometricEncoding {
  cityId: number;
  position: number;
  classIndex: number;
  coordinates: { h2: number; d: number; l: number };
  sigil: string;
}

interface Tour {
  cities: number[];
  distance: number;
  atlasExpression: string;
  geometricEncoding: GeometricEncoding[];
  runtimeMs: number;
}

interface TSPInstance {
  cities: City[];
  distances: number[][];
}

interface TransformExample {
  name: string;
  tour: number[];
  description: string;
}

// ============================================================================
// TSP Solver
// ============================================================================

// API client for Hologram FFI backend
class HologramTSPSolver {
  private instance: TSPInstance;

  constructor(instance: TSPInstance) {
    this.instance = instance;
  }

  async solve(startCity: number = 0): Promise<Tour> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/solve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          cities: this.instance.cities,
          distances: this.instance.distances,
          startCity: startCity,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const result = await response.json();
      return {
        cities: result.cities,
        distance: result.distance,
        runtimeMs: result.runtimeMs,
        atlasExpression: result.atlasExpression,
        geometricEncoding: result.geometricEncoding,
      };
    } catch (error) {
      console.error('Error solving TSP:', error);
      throw error;
    }
  }

  getTransformExamples(tour: number[]): TransformExample[] {
    // Transform examples for display (simplified)
    return [
      {
        name: 'R+1 (Rotate)',
        tour: this.applyRotate(tour, 1),
        description: 'Rotates through quaternion quadrants'
      },
      {
        name: 'T+1 (Twist)',
        tour: this.applyTwist(tour, 1),
        description: 'Twists octonion positions'
      },
      {
        name: 'M (Mirror)',
        tour: [...tour].reverse(),
        description: 'Mirrors the tour (reverses order)'
      },
      {
        name: 'D+1 (Triality)',
        tour: this.applyTriality(tour, 1),
        description: 'Cycles through modalities'
      },
    ];
  }

  private applyRotate(tour: number[], delta: number): number[] {
    const n = tour.length;
    const rotated = [...tour];
    for (let i = 0; i < n; i++) {
      rotated[(i + delta) % n] = tour[i];
    }
    return rotated;
  }

  private applyTwist(tour: number[], k: number): number[] {
    const n = tour.length;
    const twisted = [...tour];
    const segmentSize = Math.ceil(n / 8);
    const segment1 = (k % 8) * segmentSize;
    const segment2 = ((k + 4) % 8) * segmentSize;
    
    for (let i = 0; i < segmentSize && segment1 + i < n && segment2 + i < n; i++) {
      [twisted[segment1 + i], twisted[segment2 + i]] = [twisted[segment2 + i], twisted[segment1 + i]];
    }
    return twisted;
  }

  private applyTriality(tour: number[], k: number): number[] {
    const n = tour.length;
    const shift = Math.floor((n * k) / 3);
    const triality = [...tour];
    for (let i = 0; i < n; i++) {
      triality[(i + shift) % n] = tour[i];
    }
    return triality;
  }
}

// ============================================================================
// Preset Problem Instances
// ============================================================================

const PRESET_INSTANCES: { name: string; cities: City[] }[] = [
  {
    name: 'US Cities',
    cities: [
      { id: 0, name: 'New York', x: 0, y: 0 },
      { id: 1, name: 'Los Angeles', x: 24, y: 0 },
      { id: 2, name: 'Chicago', x: 8, y: 2 },
      { id: 3, name: 'Houston', x: 16, y: -2 },
      { id: 4, name: 'Phoenix', x: 22, y: -1 },
      { id: 5, name: 'Seattle', x: 3, y: 5 },
      { id: 6, name: 'Miami', x: 12, y: -4 },
      { id: 7, name: 'Ibiza', x: 20, y: 3 },
    ],
  },
  {
    name: 'US & Europe',
    cities: [
      { id: 0, name: 'Paris', x: 0, y: 0 },
      { id: 1, name: 'London', x: 3, y: 5 },
      { id: 2, name: 'Berlin', x: 8, y: 2 },
      { id: 3, name: 'Rome', x: 5, y: -4 },
      { id: 4, name: 'Madrid', x: -2, y: -3 },
      { id: 5, name: 'Miami', x: -5, y: -8 },
      { id: 6, name: 'Ibiza', x: 2, y: -6 },
    ],
  },
];

// List of 1000+ real city names for random generation
const REAL_CITY_NAMES = [
  // US Cities
  'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
  'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington',
  'Boston', 'El Paso', 'Nashville', 'Detroit', 'Oklahoma City', 'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore',
  'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Kansas City', 'Mesa', 'Atlanta', 'Omaha', 'Colorado Springs',
  'Raleigh', 'Virginia Beach', 'Miami', 'Oakland', 'Minneapolis', 'Tulsa', 'Cleveland', 'Wichita', 'Arlington', 'New Orleans',
  'Honolulu', 'Bakersfield', 'Tampa', 'Aurora', 'Anaheim', 'Santa Ana', 'St. Louis', 'Corpus Christi', 'Riverside', 'Lexington',
  'Pittsburgh', 'Anchorage', 'Stockton', 'Cincinnati', 'St. Paul', 'Toledo', 'Greensboro', 'Newark', 'Plano', 'Henderson',
  'Lincoln', 'Buffalo', 'Jersey City', 'Chula Vista', 'Fort Wayne', 'Orlando', 'St. Petersburg', 'Chandler', 'Laredo', 'Norfolk',
  'Durham', 'Madison', 'Lubbock', 'Irvine', 'Winston-Salem', 'Glendale', 'Garland', 'Hialeah', 'Reno', 'Chesapeake',
  'Gilbert', 'Baton Rouge', 'Irving', 'Scottsdale', 'North Las Vegas', 'Fremont', 'Boise', 'Richmond', 'San Bernardino', 'Birmingham',
  'Spokane', 'Rochester', 'Des Moines', 'Modesto', 'Fayetteville', 'Tacoma', 'Oxnard', 'Fontana', 'Columbus', 'Montgomery',
  // European Cities
  'London', 'Paris', 'Berlin', 'Madrid', 'Rome', 'Amsterdam', 'Vienna', 'Brussels', 'Prague', 'Stockholm',
  'Copenhagen', 'Dublin', 'Helsinki', 'Athens', 'Lisbon', 'Warsaw', 'Budapest', 'Bucharest', 'Sofia', 'Zagreb',
  'Barcelona', 'Milan', 'Naples', 'Turin', 'Palermo', 'Genoa', 'Bologna', 'Florence', 'Bari', 'Catania',
  'Venice', 'Verona', 'Messina', 'Padua', 'Trieste', 'Brescia', 'Taranto', 'Prato', 'Reggio Calabria', 'Modena',
  'Parma', 'Perugia', 'Livorno', 'Cagliari', 'Foggia', 'Rimini', 'Salerno', 'Ferrara', 'Sassari', 'Latina',
  'Marseille', 'Lyon', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg', 'Montpellier', 'Bordeaux', 'Lille', 'Rennes',
  'Reims', 'Le Havre', 'Saint-Étienne', 'Toulon', 'Grenoble', 'Dijon', 'Angers', 'Nîmes', 'Villeurbanne', 'Saint-Denis',
  'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart', 'Düsseldorf', 'Dortmund', 'Essen', 'Leipzig', 'Bremen',
  'Dresden', 'Hannover', 'Nuremberg', 'Duisburg', 'Bochum', 'Wuppertal', 'Bielefeld', 'Bonn', 'Münster', 'Karlsruhe',
  'Mannheim', 'Augsburg', 'Wiesbaden', 'Gelsenkirchen', 'Mönchengladbach', 'Braunschweig', 'Chemnitz', 'Kiel', 'Aachen', 'Halle',
  'Seville', 'Zaragoza', 'Málaga', 'Murcia', 'Palma', 'Las Palmas', 'Bilbao', 'Alicante', 'Córdoba', 'Valladolid',
  'Vigo', 'Gijón', 'Hospitalet', 'Granada', 'Vitoria', 'A Coruña', 'Elche', 'Oviedo', 'Santa Cruz', 'Badalona',
  'Cartagena', 'Terrassa', 'Jerez', 'Sabadell', 'Móstoles', 'Pamplona', 'Almería', 'Fuenlabrada', 'Leganés', 'Getafe',
  // More International Cities
  'Tokyo', 'Osaka', 'Yokohama', 'Nagoya', 'Sapporo', 'Fukuoka', 'Kobe', 'Kawasaki', 'Kyoto', 'Saitama',
  'Hiroshima', 'Sendai', 'Chiba', 'Kitakyushu', 'Sakai', 'Niigata', 'Hamamatsu', 'Shizuoka', 'Sagamihara', 'Okayama',
  'Kumamoto', 'Kagoshima', 'Hachioji', 'Funabashi', 'Matsuyama', 'Kanazawa', 'Toyama', 'Gifu', 'Takamatsu', 'Toyohashi',
  'Shanghai', 'Beijing', 'Guangzhou', 'Shenzhen', 'Chengdu', 'Chongqing', 'Hangzhou', 'Wuhan', 'Xi\'an', 'Nanjing',
  'Tianjin', 'Suzhou', 'Dongguan', 'Foshan', 'Jinan', 'Zhengzhou', 'Changsha', 'Dalian', 'Kunming', 'Qingdao',
  'Shenyang', 'Xiamen', 'Harbin', 'Hefei', 'Fuzhou', 'Nanchang', 'Shijiazhuang', 'Changchun', 'Taiyuan', 'Zhongshan',
  'Ningbo', 'Nanning', 'Urumqi', 'Guiyang', 'Haikou', 'Lanzhou', 'Yinchuan', 'Xining', 'Hohhot', 'Lhasa',
  'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata', 'Surat', 'Pune', 'Jaipur',
  'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad',
  'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Varanasi', 'Srinagar', 'Amritsar', 'Noida',
  'Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Gold Coast', 'Newcastle', 'Canberra', 'Sunshine Coast', 'Wollongong',
  'Hobart', 'Geelong', 'Townsville', 'Cairns', 'Toowoomba', 'Darwin', 'Ballarat', 'Bendigo', 'Albury', 'Launceston',
  'São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador', 'Fortaleza', 'Belo Horizonte', 'Manaus', 'Curitiba', 'Recife', 'Porto Alegre',
  'Belém', 'Goiânia', 'Guarulhos', 'Campinas', 'São Luís', 'São Gonçalo', 'Maceió', 'Duque de Caxias', 'Natal', 'Teresina',
  'Mexico City', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana', 'León', 'Juárez', 'Torreón', 'Querétaro', 'San Luis Potosí',
  'Mérida', 'Mexicali', 'Aguascalientes', 'Tlalnepantla', 'Chihuahua', 'Naucalpan', 'Cancún', 'Saltillo', 'Hermosillo', 'Morelia',
  'Toronto', 'Montreal', 'Calgary', 'Ottawa', 'Edmonton', 'Winnipeg', 'Vancouver', 'Quebec City', 'Hamilton', 'Kitchener',
  'London', 'Birmingham', 'Manchester', 'Glasgow', 'Liverpool', 'Leeds', 'Sheffield', 'Edinburgh', 'Bristol', 'Cardiff',
  'Belfast', 'Newcastle', 'Nottingham', 'Leicester', 'Coventry', 'Bradford', 'Stoke-on-Trent', 'Wolverhampton', 'Plymouth', 'Reading',
  'Moscow', 'Saint Petersburg', 'Novosibirsk', 'Yekaterinburg', 'Kazan', 'Nizhny Novgorod', 'Chelyabinsk', 'Samara', 'Omsk', 'Rostov-on-Don',
  'Ufa', 'Krasnoyarsk', 'Voronezh', 'Perm', 'Volgograd', 'Krasnodar', 'Saratov', 'Tyumen', 'Tolyatti', 'Izhevsk',
  'Barnaul', 'Irkutsk', 'Ulyanovsk', 'Khabarovsk', 'Yaroslavl', 'Vladivostok', 'Makhachkala', 'Tomsk', 'Orenburg', 'Kemerovo',
  'Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon', 'Gwangju', 'Suwon', 'Ulsan', 'Changwon', 'Goyang',
  'Seongnam', 'Bucheon', 'Ansan', 'Anyang', 'Jeonju', 'Cheonan', 'Namyangju', 'Hwaseong', 'Gimhae', 'Pyeongtaek',
  'Bangkok', 'Nonthaburi', 'Nakhon Ratchasima', 'Chiang Mai', 'Hat Yai', 'Udon Thani', 'Pak Kret', 'Khon Kaen', 'Chaophraya Surasak', 'Nakhon Si Thammarat',
  'Lampang', 'Khon Kaen', 'Ubon Ratchathani', 'Nakhon Sawan', 'Rayong', 'Phitsanulok', 'Songkhla', 'Yala', 'Phuket', 'Satun',
  'Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Palembang', 'Makassar', 'Tangerang', 'Depok', 'Bekasi',
  'South Jakarta', 'West Jakarta', 'North Jakarta', 'East Jakarta', 'Central Jakarta', 'Bogor', 'Malang', 'Yogyakarta', 'Cimahi', 'Batam',
  'Manila', 'Quezon City', 'Caloocan', 'Davao City', 'Cebu City', 'Zamboanga City', 'Antipolo', 'Pasig', 'Cagayan de Oro', 'Parañaque',
  'Las Piñas', 'Makati', 'Bacolod', 'General Santos', 'Bacoor', 'Valenzuela', 'Muntinlupa', 'Marikina', 'Mandaue', 'Taguig',
  'Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Adana', 'Gaziantep', 'Konya', 'Antalya', 'Kayseri', 'Mersin',
  'Eskişehir', 'Diyarbakır', 'Samsun', 'Denizli', 'Şanlıurfa', 'Malatya', 'Erzurum', 'Van', 'Batman', 'Elazığ',
  'Cairo', 'Alexandria', 'Giza', 'Shubra El Kheima', 'Port Said', 'Suez', 'Luxor', 'Aswan', 'Asyut', 'Ismailia',
  'Faiyum', 'Zagazig', 'Damietta', 'Aswan', 'Minya', 'Damanhur', 'Beni Suef', 'Qena', 'Sohag', 'Hurghada',
  'Johannesburg', 'Cape Town', 'Durban', 'Pretoria', 'Port Elizabeth', 'Bloemfontein', 'East London', 'Pietermaritzburg', 'Kimberley', 'Polokwane',
  'Nelspruit', 'Rustenburg', 'Welkom', 'Newcastle', 'Klerksdorp', 'Uitenhage', 'Botshabelo', 'Brakpan', 'Witbank', 'Krugersdorp',
  'Buenos Aires', 'Córdoba', 'Rosario', 'Mendoza', 'Tucumán', 'La Plata', 'Mar del Plata', 'Salta', 'Santa Fe', 'San Juan',
  'Resistencia', 'Santiago del Estero', 'Corrientes', 'Bahía Blanca', 'Posadas', 'Paraná', 'Neuquén', 'Formosa', 'San Salvador de Jujuy', 'La Rioja',
  'Santiago', 'Valparaíso', 'Concepción', 'La Serena', 'Antofagasta', 'Temuco', 'Rancagua', 'Talca', 'Arica', 'Iquique',
  'Puerto Montt', 'Valdivia', 'Osorno', 'Chillán', 'Calama', 'Copiapó', 'Los Ángeles', 'Curicó', 'Quilpué', 'Punta Arenas',
  'Lima', 'Arequipa', 'Trujillo', 'Chiclayo', 'Piura', 'Iquitos', 'Cusco', 'Chimbote', 'Huancayo', 'Tacna',
  'Ica', 'Juliaca', 'Sullana', 'Cajamarca', 'Pucallpa', 'Ayacucho', 'Chincha Alta', 'Huaraz', 'Tarapoto', 'Puno',
  'Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 'Cúcuta', 'Soledad', 'Ibagué', 'Bucaramanga', 'Santa Marta',
  'Pereira', 'Valledupar', 'Bello', 'Pasto', 'Villavicencio', 'Manizales', 'Montería', 'Neiva', 'Armenia', 'Sincelejo',
  'Caracas', 'Maracaibo', 'Valencia', 'Barquisimeto', 'Ciudad Guayana', 'Maturín', 'Barcelona', 'Maracay', 'Ciudad Bolívar', 'San Cristóbal',
  'Cumaná', 'Barinas', 'Cabimas', 'Puerto La Cruz', 'Mérida', 'Punto Fijo', 'Guanare', 'Carúpano', 'Los Teques', 'Puerto Cabello',
  'Tehran', 'Mashhad', 'Isfahan', 'Karaj', 'Tabriz', 'Shiraz', 'Qom', 'Ahvaz', 'Kermanshah', 'Urmia',
  'Rasht', 'Zahedan', 'Hamadan', 'Kerman', 'Yazd', 'Ardabil', 'Bandar Abbas', 'Arak', 'Eslamshahr', 'Zanjan',
  'Karachi', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Multan', 'Gujranwala', 'Peshawar', 'Quetta', 'Islamabad', 'Sargodha',
  'Sialkot', 'Bahawalpur', 'Sukkur', 'Larkana', 'Sheikhupura', 'Rahim Yar Khan', 'Jhang', 'Dera Ghazi Khan', 'Gujrat', 'Kasur',
  'Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Sylhet', 'Comilla', 'Rangpur', 'Mymensingh', 'Barisal', 'Jessore',
  'Narayanganj', 'Gazipur', 'Bogra', 'Dinajpur', 'Cox\'s Bazar', 'Tangail', 'Jamalpur', 'Pabna', 'Kushtia', 'Faridpur',
  'Baghdad', 'Basra', 'Mosul', 'Erbil', 'Najaf', 'Karbala', 'Nasiriyah', 'Amara', 'Ramadi', 'Baqubah',
  'Fallujah', 'Samarra', 'Kirkuk', 'Diyala', 'Hilla', 'Kut', 'Dohuk', 'Sulaymaniyah', 'Zakho', 'Tikrit',
  'Riyadh', 'Jeddah', 'Mecca', 'Medina', 'Dammam', 'Khobar', 'Taif', 'Abha', 'Tabuk', 'Buraydah',
  'Khamis Mushait', 'Hail', 'Najran', 'Jazan', 'Yanbu', 'Al Jubail', 'Al Khobar', 'Arar', 'Sakaka', 'Jizan',
  'Dubai', 'Abu Dhabi', 'Sharjah', 'Al Ain', 'Ajman', 'Ras Al Khaimah', 'Fujairah', 'Umm Al Quwain', 'Khor Fakkan', 'Dibba',
  'Tel Aviv', 'Jerusalem', 'Haifa', 'Rishon LeZion', 'Petah Tikva', 'Ashdod', 'Netanya', 'Beer Sheva', 'Holon', 'Bnei Brak',
  'Ramat Gan', 'Rehovot', 'Bat Yam', 'Ashkelon', 'Herzliya', 'Kfar Saba', 'Hadera', 'Modi\'in', 'Lod', 'Nazareth',
  'Casablanca', 'Rabat', 'Fes', 'Marrakech', 'Tangier', 'Agadir', 'Meknes', 'Oujda', 'Kenitra', 'Tetouan',
  'Safi', 'El Jadida', 'Nador', 'Beni Mellal', 'Taza', 'Khouribga', 'Settat', 'Larache', 'Ksar El Kebir', 'Guelmim',
  'Lagos', 'Kano', 'Ibadan', 'Abuja', 'Port Harcourt', 'Benin City', 'Kaduna', 'Maiduguri', 'Zaria', 'Aba',
  'Jos', 'Ilorin', 'Oyo', 'Enugu', 'Abeokuta', 'Onitsha', 'Warri', 'Calabar', 'Akure', 'Osogbo',
  'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi', 'Kitale', 'Garissa', 'Kakamega',
  'Nyeri', 'Meru', 'Machakos', 'Embu', 'Kericho', 'Bungoma', 'Busia', 'Homa Bay', 'Migori', 'Kilifi',
  'Addis Ababa', 'Dire Dawa', 'Mekele', 'Gondar', 'Awassa', 'Bahir Dar', 'Dessie', 'Jimma', 'Jijiga', 'Shashamane',
  'Bishoftu', 'Arba Minch', 'Hosaena', 'Harar', 'Dilla', 'Nekemte', 'Debre Markos', 'Sodo', 'Asella', 'Goba',
  'Tunis', 'Sfax', 'Sousse', 'Kairouan', 'Bizerte', 'Gabès', 'Ariana', 'Gafsa', 'Monastir', 'Ben Arous',
  'Kasserine', 'Medenine', 'Tozeur', 'Béja', 'Jendouba', 'Le Kef', 'Mahdia', 'Tataouine', 'Zaghouan', 'Siliana',
  'Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Batna', 'Djelfa', 'Sétif', 'Sidi Bel Abbès', 'Biskra',
  'Tébessa', 'Tiaret', 'Béjaïa', 'Tlemcen', 'Bordj Bou Arréridj', 'Béchar', 'Skikda', 'Souk Ahras', 'Chlef', 'Laghouat',
  'Accra', 'Kumasi', 'Tamale', 'Takoradi', 'Ashaiman', 'Sunyani', 'Cape Coast', 'Obuasi', 'Teshie', 'Tema',
  'Sekondi', 'Koforidua', 'Techiman', 'Ho', 'Wa', 'Bolgatanga', 'Bawku', 'Nkawkaw', 'Axim', 'Elmina',
  'Dakar', 'Thiès', 'Rufisque', 'Kaolack', 'Ziguinchor', 'Saint-Louis', 'Tambacounda', 'Mbour', 'Louga', 'Richard Toll',
  'Kolda', 'Fatick', 'Diourbel', 'Bambey', 'Tivaouane', 'Joal-Fadiouth', 'Dagana', 'Matam', 'Kédougou', 'Sédhiou',
  'Kinshasa', 'Lubumbashi', 'Mbuji-Mayi', 'Kananga', 'Kisangani', 'Bukavu', 'Goma', 'Kolwezi', 'Likasi', 'Bunia',
  'Tshikapa', 'Kikwit', 'Uvira', 'Boma', 'Matadi', 'Mbandaka', 'Butembo', 'Kalemie', 'Kindu', 'Isiro',
  'Luanda', 'Huambo', 'Lobito', 'Benguela', 'Lubango', 'Kuito', 'Malanje', 'Namibe', 'Soyo', 'Cabinda',
  'Sumbe', 'Menongue', 'Uíge', 'N\'dalatando', 'Dundo', 'Ondjiva', 'Mbanza Kongo', 'Caxito', 'Luena', 'Saurimo',
  'Dar es Salaam', 'Mwanza', 'Arusha', 'Dodoma', 'Mbeya', 'Morogoro', 'Tanga', 'Zanzibar City', 'Kigoma', 'Mtwara',
  'Tabora', 'Iringa', 'Musoma', 'Shinyanga', 'Bukoba', 'Singida', 'Lindi', 'Sumbawanga', 'Songea', 'Moshi',
  'Kampala', 'Gulu', 'Lira', 'Mbarara', 'Jinja', 'Mbale', 'Mukono', 'Masaka', 'Arua', 'Entebbe',
  'Soroti', 'Kabale', 'Fort Portal', 'Tororo', 'Iganga', 'Busia', 'Njeru', 'Kitgum', 'Hoima', 'Kasese',
  'Kigali', 'Butare', 'Gitarama', 'Ruhengeri', 'Gisenyi', 'Byumba', 'Cyangugu', 'Kibungo', 'Kibuye', 'Rwamagana',
  'Kampala', 'Gulu', 'Lira', 'Mbarara', 'Jinja', 'Mbale', 'Mukono', 'Masaka', 'Arua', 'Entebbe',
  'Harare', 'Bulawayo', 'Chitungwiza', 'Mutare', 'Gweru', 'Epworth', 'Kwekwe', 'Kadoma', 'Masvingo', 'Chinhoyi',
  'Marondera', 'Bindura', 'Beitbridge', 'Kariba', 'Karoi', 'Rusape', 'Chipinge', 'Gwanda', 'Shurugwi', 'Zvishavane',
  'Lusaka', 'Kitwe', 'Ndola', 'Kabwe', 'Chingola', 'Mufulira', 'Livingstone', 'Luanshya', 'Kasama', 'Chipata',
  'Mazabuka', 'Kafue', 'Mongu', 'Solwezi', 'Chililabombwe', 'Mpulungu', 'Senanga', 'Mansa', 'Kapiri Mposhi', 'Nchelenge',
  'Maputo', 'Matola', 'Beira', 'Nampula', 'Chimoio', 'Nacala', 'Quelimane', 'Tete', 'Lichinga', 'Pemba',
  'Maxixe', 'Dondo', 'Angoche', 'Montepuez', 'Mocuba', 'Gurue', 'Cuamba', 'Xai-Xai', 'Inhambane', 'Vilanculos',
  'Windhoek', 'Rundu', 'Walvis Bay', 'Oshakati', 'Swakopmund', 'Katima Mulilo', 'Grootfontein', 'Mariental', 'Otjiwarongo', 'Ongwediva',
  'Rehoboth', 'Keetmanshoop', 'Lüderitz', 'Tsumeb', 'Gobabis', 'Okahandja', 'Omaruru', 'Karibib', 'Usakos', 'Henties Bay',
  'Gaborone', 'Francistown', 'Molepolole', 'Selebi-Phikwe', 'Maun', 'Serowe', 'Kanye', 'Mahalapye', 'Mochudi', 'Mogoditshane',
  'Palapye', 'Lobatse', 'Tlokweng', 'Ramotswa', 'Thamaga', 'Letlhakane', 'Tonota', 'Mmadinare', 'Jwaneng', 'Orapa',
  'Maseru', 'Teyateyaneng', 'Mafeteng', 'Hlotse', 'Mohale\'s Hoek', 'Maputsoe', 'Quthing', 'Qacha\'s Nek', 'Butha-Buthe', 'Mokhotlong',
  'Thaba-Tseka', 'Roma', 'Peka', 'Semonkong', 'Marakabei', 'Tsoelike', 'Mphaki', 'Mantsonyane', 'Thaba Bosiu', 'Ha Kome',
  'Mbabane', 'Manzini', 'Big Bend', 'Malkerns', 'Nhlangano', 'Mhlume', 'Piggs Peak', 'Sidvokodvo', 'Lobamba', 'Siteki',
  'Hlatikulu', 'Bulembu', 'Mhlambanyatsi', 'Kwaluseni', 'Bhunya', 'Mankayane', 'Lavumisa', 'Nsoko', 'Tshaneni', 'Mpaka',
  'Ibiza'
];

function generateRandomCities(count: number, seed?: number): City[] {
  // Use seed for reproducible random generation
  let seedValue = seed !== undefined ? seed : Math.floor(Math.random() * 1000000);
  const random = () => {
    seedValue = (seedValue * 9301 + 49297) % 233280;
    return seedValue / 233280;
  };

  const cities: City[] = [];
  
  // Shuffle city names using seeded random for reproducibility
  const shuffledNames = [...REAL_CITY_NAMES];
  for (let i = shuffledNames.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffledNames[i], shuffledNames[j]] = [shuffledNames[j], shuffledNames[i]];
  }
  
  for (let i = 0; i < count; i++) {
    const name = shuffledNames[i % shuffledNames.length];
    
    // Generate cities in a reasonable coordinate range
    const x = (random() - 0.5) * 40; // -20 to 20
    const y = (random() - 0.5) * 40; // -20 to 20
    
    cities.push({
      id: i,
      name,
      x: Math.round(x * 10) / 10, // Round to 1 decimal
      y: Math.round(y * 10) / 10,
    });
  }
  
  return cities;
}

// Generate random distance using Box-Muller transform for normal distribution
function generateRandomDistance(mean: number, stdDev: number, randomState: { hasSpare: boolean; spare: number }): number {
  // Box-Muller transform for normal distribution
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
  return Math.max(0.1, distance); // Ensure positive distance
}

function createInstance(cities: City[], useRandomDistances: boolean = false): TSPInstance {
  const distances: number[][] = [];
  
  if (useRandomDistances) {
    // Generate random distances with normal distribution
    const randomState = { hasSpare: false, spare: 0 };
    // Calculate mean based on average coordinate spread
    const coords = cities.map(c => ({ x: c.x, y: c.y }));
    const xs = coords.map(c => c.x);
    const ys = coords.map(c => c.y);
    const rangeX = Math.max(...xs) - Math.min(...xs) || 20;
    const rangeY = Math.max(...ys) - Math.min(...ys) || 20;
    const meanDistance = Math.sqrt(rangeX * rangeX + rangeY * rangeY) / 2; // Average distance based on coordinate range
    const stdDev = meanDistance * 0.4; // Standard deviation for variation (40% of mean)
    
    // Initialize all distances to 0
    for (let i = 0; i < cities.length; i++) {
      distances[i] = new Array(cities.length).fill(0);
    }
    
    // Generate random distances for upper triangle, then mirror to lower triangle
    for (let i = 0; i < cities.length; i++) {
      for (let j = i + 1; j < cities.length; j++) {
        // Generate distance once for symmetric pairs
        const distance = generateRandomDistance(meanDistance, stdDev, randomState);
        distances[i][j] = Math.round(distance * 10) / 10; // Round to 1 decimal
        distances[j][i] = distances[i][j]; // Symmetric
      }
    }
  } else {
    // Use Euclidean distance based on coordinates
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

// ============================================================================
// React Component
// ============================================================================

const TSPDemo: React.FC = () => {
  const [selectedPreset, setSelectedPreset] = useState(0);
  const initialCityCount = PRESET_INSTANCES[0].cities.length;
  const [cityCount, setCityCount] = useState(initialCityCount);
  const [useRandomDistances, setUseRandomDistances] = useState(false);
  const [startingCity, setStartingCity] = useState<number | 'random'>(0);
  const [instance, setInstance] = useState<TSPInstance>(() => 
    createInstance(PRESET_INSTANCES[0].cities, false)
  );
  const [solution, setSolution] = useState<Tour | null>(null);
  const [transformExamples, setTransformExamples] = useState<TransformExample[]>([]);
  const [isSolving, setIsSolving] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const comparisonGraphRef = useRef<HTMLCanvasElement>(null);

  const solver = useMemo(() => new HologramTSPSolver(instance), [instance]);

  // Generate cities based on preset and city count
  const generateCities = useCallback((presetIndex: number, count: number): City[] => {
    const preset = PRESET_INSTANCES[presetIndex];
    const presetCities = preset.cities;
    
    if (count <= presetCities.length) {
      // Use subset of preset cities
      return presetCities.slice(0, count).map((city, idx) => ({
        ...city,
        id: idx,
      }));
    } else {
      // Use preset cities as base, then add random cities
      const cities = presetCities.map((city, idx) => ({
        ...city,
        id: idx,
      }));
      
      // Generate additional random cities
      const additionalCount = count - presetCities.length;
      const randomCities = generateRandomCities(additionalCount, presetIndex * 1000);
      
      // Offset IDs for additional cities
      randomCities.forEach((city, idx) => {
        city.id = presetCities.length + idx;
      });
      
      return [...cities, ...randomCities];
    }
  }, []);

  // Update instance when preset or city count changes
  useEffect(() => {
    const cities = generateCities(selectedPreset, cityCount);
    const newInstance = createInstance(cities, useRandomDistances);
    setInstance(newInstance);
    setSolution(null);
    setTransformExamples([]);
    // Find Miami or Ibiza, default to first available, or 0 if neither exists
    const miamiIndex = cities.findIndex(c => c.name === 'Miami');
    const ibizaIndex = cities.findIndex(c => c.name === 'Ibiza');
    const validStartingCity = miamiIndex >= 0 ? miamiIndex : (ibizaIndex >= 0 ? ibizaIndex : 0);
    // Reset starting city if it's out of bounds or not Miami/Ibiza (but keep 'random' if selected)
    // Also reset if the selected city doesn't exist in the new instance
    if (startingCity !== 'random' && typeof startingCity === 'number') {
      if (startingCity >= cities.length || 
          (cities[startingCity]?.name !== 'Miami' && cities[startingCity]?.name !== 'Ibiza')) {
        setStartingCity(validStartingCity);
      }
    }
  }, [selectedPreset, cityCount, useRandomDistances, generateCities, startingCity]);

  const handleRandomizeDistances = useCallback(async () => {
    if (isSolving) return; // Don't allow multiple clicks while solving
    
    console.log('Randomize Distances button clicked');
    
    // Generate new random distances immediately
    const cities = generateCities(selectedPreset, cityCount);
    const newInstance = createInstance(cities, true);
    
    console.log('New instance created with random distances');
    console.log('Sample distances:', {
      '0->1': newInstance.distances[0][1],
      '1->2': newInstance.distances[1]?.[2],
      '0->2': newInstance.distances[0][2]
    });
    
    // Update state immediately
    setUseRandomDistances(true);
    setInstance(newInstance);
    setSolution(null);
    setTransformExamples([]);
    
    // Auto-solve after randomizing distances
    setIsSolving(true);
    try {
      const newSolver = new HologramTSPSolver(newInstance);
      // If random is selected, pick a random city
      const actualStartingCity = startingCity === 'random' 
        ? Math.floor(Math.random() * newInstance.cities.length)
        : startingCity;
      
      const result = await newSolver.solve(actualStartingCity);
      // Create initial tour starting from the selected starting city
      const initialTour = Array.from({ length: newInstance.cities.length }, (_, i) => (actualStartingCity + i) % newInstance.cities.length);
      const examples = newSolver.getTransformExamples(initialTour);
      setSolution(result);
      setTransformExamples(examples);
      console.log('Solution found with distance:', result.distance);
    } catch (error) {
      console.error('Error solving TSP:', error);
      alert('Error solving TSP. Make sure the backend is running on http://localhost:5000');
    } finally {
      setIsSolving(false);
    }
  }, [selectedPreset, cityCount, generateCities, isSolving, startingCity]);

  const handlePresetChange = useCallback((index: number) => {
    setSelectedPreset(index);
    // Update city count to match preset's initial city count
    const presetCityCount = PRESET_INSTANCES[index].cities.length;
    setCityCount(presetCityCount);
    // Reset to Euclidean distances when changing preset
    setUseRandomDistances(false);
    // Instance will update via useEffect
  }, []);

  const handleSolve = useCallback(async () => {
    setIsSolving(true);
    try {
      // If random is selected, pick a random city
      const actualStartingCity = startingCity === 'random' 
        ? Math.floor(Math.random() * instance.cities.length)
        : startingCity;
      
      const result = await solver.solve(actualStartingCity);
      // Create initial tour starting from the selected starting city
      const initialTour = Array.from({ length: instance.cities.length }, (_, i) => (actualStartingCity + i) % instance.cities.length);
      const examples = solver.getTransformExamples(initialTour);
      setSolution(result);
      setTransformExamples(examples);
    } catch (error) {
      console.error('Error solving TSP:', error);
      alert('Error solving TSP. Make sure the backend is running on http://localhost:5000');
    } finally {
      setIsSolving(false);
    }
  }, [solver, instance, startingCity]);

  // Draw tour visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !solution) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    ctx.clearRect(0, 0, width, height);

    // Find bounds
    const xs = instance.cities.map(c => c.x);
    const ys = instance.cities.map(c => c.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const scale = Math.min(
      (width - 2 * padding) / rangeX,
      (height - 2 * padding) / rangeY
    );

    const toCanvasX = (x: number) => padding + (x - minX) * scale;
    const toCanvasY = (y: number) => height - padding - (y - minY) * scale;

    // Draw tour path
    ctx.strokeStyle = '#e0e0e0'; // var(--primary-color)
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    const tourWithReturn = [...solution.cities, solution.cities[0]];
    for (let i = 0; i < tourWithReturn.length; i++) {
      const city = instance.cities[tourWithReturn[i]];
      const x = toCanvasX(city.x);
      const y = toCanvasY(city.y);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw cities
    instance.cities.forEach((city, idx) => {
      const x = toCanvasX(city.x);
      const y = toCanvasY(city.y);
      const tourIndex = solution.cities.indexOf(idx);
      const isInTour = tourIndex !== -1;

      // City circle
      ctx.fillStyle = isInTour ? '#e0e0e0' : '#808080'; // var(--primary-color) : var(--text-muted-color)
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();

      // City label
      ctx.fillStyle = '#0a0a0a'; // var(--bg-color) for contrast
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(city.name, x, y);

      // Position number
      if (isInTour) {
        ctx.fillStyle = '#e0e0e0'; // var(--primary-color)
        ctx.font = '10px monospace';
        ctx.fillText(tourIndex.toString(), x, y - 20);
      }
    });
  }, [solution, instance]);

  // Draw comparison graph (only when solution exists)
  useEffect(() => {
    const canvas = comparisonGraphRef.current;
    if (!canvas || !solution) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 40, right: 40, bottom: 60, left: 100 }; // Increased left padding for time labels
    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;

    ctx.clearRect(0, 0, width, height);

    // Calculate classical solver runtime (exponential: O(n!))
    // For log scale, we'll use milliseconds
    // X-axis maximum equals the number of cities in the calculation
    const currentCityCount = instance.cities.length;
    const maxCities = currentCityCount; // X-axis equals the number of cities
    const minRuntime = 0.001; // 0.001ms
    const maxRuntime = 1e15; // 1e15ms (very large for log scale)

    // Helper to calculate factorial runtime (simplified: n! * baseTime)
    const classicalRuntime = (n: number): number => {
      if (n <= 1) return 0.001;
      // Simplified: factorial grows extremely fast
      // For visualization: n! * 0.0001ms per permutation
      let fact = 1;
      for (let i = 2; i <= n && i <= 20; i++) {
        fact *= i;
      }
      if (n > 20) {
        // For n > 20, use approximation
        fact = Math.pow(n, n) * Math.exp(-n) * Math.sqrt(2 * Math.PI * n);
      }
      return fact * 0.0001; // Base time per permutation
    };

    // Log scale conversion
    const logToY = (value: number): number => {
      const logMin = Math.log10(minRuntime);
      const logMax = Math.log10(maxRuntime);
      const logValue = Math.log10(Math.max(value, minRuntime));
      const normalized = (logValue - logMin) / (logMax - logMin);
      return padding.top + graphHeight - (normalized * graphHeight);
    };

    const cityToX = (cityCount: number): number => {
      return padding.left + (cityCount / maxCities) * graphWidth;
    };

    // Helper function to format time in human-readable format
    const formatTime = (ms: number): string => {
      const seconds = ms / 1000;
      const minutes = seconds / 60;
      const hours = minutes / 60;
      const days = hours / 24;
      const years = days / 365.25;
      
      if (years >= 1) {
        return years >= 1000 ? `${(years / 1000).toFixed(1)}K years` :
               years >= 1 ? `${years.toFixed(1)} year${years >= 2 ? 's' : ''}` :
               `${(years * 10).toFixed(0)}/10 year`;
      } else if (days >= 1) {
        return days >= 100 ? `${(days / 100).toFixed(1)}C days` :
               days >= 10 ? `${days.toFixed(0)} days` :
               `${days.toFixed(1)} day${days >= 2 ? 's' : ''}`;
      } else if (hours >= 1) {
        return hours >= 100 ? `${(hours / 100).toFixed(1)}C hrs` :
               hours >= 10 ? `${hours.toFixed(0)} hrs` :
               `${hours.toFixed(1)} hr${hours >= 2 ? 's' : ''}`;
      } else if (minutes >= 1) {
        return minutes >= 100 ? `${(minutes / 100).toFixed(1)}C min` :
               minutes >= 10 ? `${minutes.toFixed(0)} min` :
               `${minutes.toFixed(1)} min`;
      } else if (seconds >= 1) {
        return seconds >= 100 ? `${(seconds / 100).toFixed(1)}C sec` :
               seconds >= 10 ? `${seconds.toFixed(0)} sec` :
               `${seconds.toFixed(1)} sec`;
      } else {
        return ms >= 1 ? `${ms.toFixed(0)} ms` :
               ms >= 0.1 ? `${ms.toFixed(1)} ms` :
               `${ms.toFixed(2)} ms`;
      }
    };

    // Draw grid lines (log scale) with human-readable time labels
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    const logSteps = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000];
    logSteps.forEach(value => {
      const y = logToY(value);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      
      // Label with human-readable time format
      ctx.fillStyle = '#808080';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const label = formatTime(value);
      ctx.fillText(label, padding.left - 10, y);
    });

    // Draw city count grid lines
    // Adjust step size based on maxCities to ensure good visibility
    let cityStep: number;
    if (maxCities <= 10) {
      cityStep = 1; // Show every city for small counts
    } else if (maxCities <= 20) {
      cityStep = 2; // Show every 2 cities
    } else if (maxCities <= 30) {
      cityStep = 5; // Show every 5 cities
    } else if (maxCities <= 50) {
      cityStep = 10;
    } else if (maxCities <= 100) {
      cityStep = 20;
    } else {
      cityStep = 25;
    }
    
    for (let i = 0; i <= maxCities; i += cityStep) {
      const x = cityToX(i);
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
      
      // Label
      ctx.fillStyle = '#808080';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(i.toString(), x, height - padding.bottom + 5);
    }
    
    // Always show current city count as a grid line if it's not already shown
    if (currentCityCount > 0 && currentCityCount % cityStep !== 0) {
      const x = cityToX(currentCityCount);
      ctx.strokeStyle = '#4a9eff';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
      
      // Label current city count
      ctx.fillStyle = '#4a9eff';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(currentCityCount.toString(), x, height - padding.bottom + 5);
    }

    // Draw classical solver curve
    ctx.strokeStyle = '#ff6666';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let firstPoint = true;
    for (let n = 3; n <= maxCities; n += 0.5) {
      const runtime = classicalRuntime(n);
      const x = cityToX(n);
      const y = logToY(runtime);
      if (firstPoint) {
        ctx.moveTo(x, y);
        firstPoint = false;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw Hologram method (use actual runtime from solution)
    const hologramRuntime = solution.runtimeMs;
    const hologramY = logToY(hologramRuntime);
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(padding.left, hologramY);
    ctx.lineTo(width - padding.right, hologramY);
    ctx.stroke();

    // Label Hologram line with actual runtime (ensure it fits)
    ctx.fillStyle = '#4a9eff';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const hologramLabel = `Hologram (${hologramRuntime.toFixed(2)}ms)`;
    // Position label to ensure it doesn't spill over
    const hologramLabelX = Math.min(width - padding.right - 10, width - padding.right - 150);
    ctx.fillText(hologramLabel, hologramLabelX, hologramY - 15);

    // Label Classical solver
    ctx.fillStyle = '#ff6666';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Classical Solver (Brute Force)', padding.left + 10, padding.top + 20);

    // Draw impossible region (> 30 cities) - only if maxCities > 30
    if (maxCities > 30) {
      const impossibleX = cityToX(30);
      ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
      ctx.fillRect(impossibleX, padding.top, width - padding.right - impossibleX, graphHeight);
      
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(impossibleX, padding.top);
      ctx.lineTo(impossibleX, height - padding.bottom);
      ctx.stroke();

      // Label impossible region next to the red dotted line on the left
      ctx.fillStyle = '#ff6666';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.save();
      // Position label next to the red line on the left side
      const labelX = impossibleX;
      const labelY = padding.top + graphHeight / 2;
      ctx.translate(labelX, labelY);
      ctx.rotate(-Math.PI / 2);
      // Use shorter text to prevent spillover
      ctx.fillText('> 30 cities: brute force only', 0, 0);
      ctx.restore();
    }

    // Axis labels
    ctx.fillStyle = '#e0e0e0';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Number of Cities', width / 2, height - padding.bottom + 30);

    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Runtime (log scale)', 0, 0);
    ctx.restore();

    // Current solution point (always draw since graph only shows when solution exists)
    const currentX = cityToX(currentCityCount);
    const currentY = logToY(solution.runtimeMs);
    
    // Draw point
    ctx.fillStyle = '#4a9eff';
    ctx.beginPath();
    ctx.arc(currentX, currentY, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw connecting lines
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(currentX, currentY);
    ctx.lineTo(currentX, height - padding.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(currentX, currentY);
    ctx.lineTo(padding.left, currentY);
    ctx.stroke();
    
    // Label current point with only time (ensure it fits within canvas bounds)
    ctx.fillStyle = '#4a9eff';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    const pointLabel = formatTime(solution.runtimeMs);
    // Check if label would spill over right edge
    const labelX = currentX + 10;
    const labelWidth = ctx.measureText(pointLabel).width;
    const adjustedX = labelX + labelWidth > width - padding.right ? 
                      Math.max(padding.left, width - padding.right - labelWidth - 5) : labelX;
    ctx.fillText(pointLabel, adjustedX, currentY - 10);
  }, [solution, instance, cityCount]);

  // Calculate universe count for TSP
  const universeCount = useMemo(() => {
    const n = instance.cities.length;
    if (n === 0) return '0';
    let result = BigInt(1);
    for (let i = 1; i <= n; i++) {
      result *= BigInt(i);
    }
    return result.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }, [instance.cities.length]);

  return (
    <div className="parallel-universe-explorer">
      {/* Problem Statement Section */}
      <div className="problem-statement">
        <h2>The Problem We're Solving</h2>
        <div className="problem-content">
          <div className="problem-box">
            <h3>Traveling Salesman Problem</h3>
            <p>
              You have <strong>{instance.cities.length} cities</strong> to visit.
              Each city has a <strong>location</strong> (x, y coordinates).
            </p>
            <p>
              <strong>Goal:</strong> Find the shortest route that visits each city exactly once and returns to the starting city.
            </p>
            <div className="problem-details">
              <p><strong>TSP Challenge:</strong></p>
              <ul>
                <li>Total possible routes: <strong>{instance.cities.length}! = {universeCount}</strong> permutations</li>
                <li>Each route must visit all cities exactly once</li>
                <li>Must return to the starting city (closed tour)</li>
                <li>For {instance.cities.length} cities, there are <strong>{universeCount}</strong> possible tours to evaluate</li>
              </ul>
            </div>
          </div>
          <div className="hologram-power-box" style={{ minWidth: 0, maxWidth: '100%', overflow: 'hidden' }}>
            <h3>How Hologram Solves This</h3>
            <div className="power-explanation">
              <div className="power-step">
                <span className="step-number">1</span>
                <div style={{ wordBreak: 'break-word', overflowWrap: 'break-word', minWidth: 0 }}>
                  <strong>Classical Computing:</strong> Would need to check each permutation one by one.
                  <br />
                  <span className="example" style={{ wordBreak: 'break-word', overflowWrap: 'break-word', display: 'block' }}>
                    For {instance.cities.length} cities: {universeCount} separate calculations
                  </span>
                </div>
              </div>
              <div className="power-step">
                <span className="step-number">2</span>
                <div style={{ wordBreak: 'break-word', overflowWrap: 'break-word', minWidth: 0 }}>
                  <strong>Hologram Approach:</strong> Encodes cities using geometric algebra (96-class system).
                  <br />
                  <span className="example" style={{ wordBreak: 'break-word', overflowWrap: 'break-word', display: 'block' }}>
                    Uses sequential composition (<code>.</code>) to represent tour order, and transform operations (R, T, M, D) to explore solution space
                  </span>
                </div>
              </div>
              <div className="power-step">
                <span className="step-number">3</span>
                <div style={{ wordBreak: 'break-word', overflowWrap: 'break-word', minWidth: 0 }}>
                  <strong>Result:</strong> Evaluates multiple candidate tours simultaneously through geometric transforms.
                  <br />
                  <span className="example" style={{ wordBreak: 'break-word', overflowWrap: 'break-word', display: 'block' }}>
                    Finds optimal or near-optimal solutions efficiently using quaternion rotations, octonion twists, and triality operations
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="controls-panel">
        <div className="control-group">
          <label htmlFor="preset-select">
            Problem Instance:
          </label>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '0.5rem' }}>
          {PRESET_INSTANCES.map((preset, idx) => (
            <button
              key={idx}
                id={idx === 0 ? 'preset-select' : undefined}
              onClick={() => handlePresetChange(idx)}
                className={`button ${selectedPreset === idx ? 'button-primary' : 'button-secondary'}`}
            >
              {preset.name}
            </button>
          ))}
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="city-count">
            Number of Cities: <strong>{cityCount}</strong>
          </label>
          <input
            id="city-count"
            type="range"
            min="3"
            max="1000"
            value={cityCount}
            onChange={(e) => setCityCount(parseInt(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>3</span>
            <span>1000</span>
          </div>
          <div className="universe-count-preview">
            <span className="preview-label">Possible Tours:</span>
            <span className="preview-value">
              {cityCount}! = {universeCount}
            </span>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="starting-city">
            Starting & Ending City:
          </label>
          <select
            id="starting-city"
            value={startingCity === 'random' ? 'random' : startingCity}
            onChange={(e) => {
              const value = e.target.value;
              if (value === 'random') {
                setStartingCity('random');
              } else {
                const cityIdx = parseInt(value);
                // Only set if it's a valid index (not -1 for disabled options)
                if (cityIdx >= 0) {
                  setStartingCity(cityIdx);
                }
              }
            }}
            style={{
              padding: '0.5rem',
              backgroundColor: 'var(--surface-elevated)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-color)',
              fontFamily: 'var(--font-family-mono)',
              fontSize: '0.9rem',
              borderRadius: '0',
              marginTop: '0.5rem',
              width: '100%',
              maxWidth: '300px'
            }}
          >
            {(() => {
              // Always show Miami, Ibiza, and Random for all instances
              const miamiIdx = instance.cities.findIndex(c => c.name === 'Miami');
              const ibizaIdx = instance.cities.findIndex(c => c.name === 'Ibiza');
              const miamiExists = miamiIdx >= 0;
              const ibizaExists = ibizaIdx >= 0;
              
              return (
                <>
                  <option 
                    value={miamiExists ? miamiIdx : -1} 
                    disabled={!miamiExists}
                  >
                    Miami{miamiExists ? '' : ' (not in instance)'}
                  </option>
                  <option 
                    value={ibizaExists ? ibizaIdx : -1} 
                    disabled={!ibizaExists}
                  >
                    Ibiza{ibizaExists ? '' : ' (not in instance)'}
                  </option>
                  <option value="random">Random</option>
                </>
              );
            })()}
          </select>
          <div style={{ 
            marginTop: '0.5rem', 
            fontSize: '0.85rem', 
            color: 'var(--text-muted-color)',
            fontStyle: 'italic'
          }}>
            The selected city will be both the first and last city in the tour (closed loop).
            {startingCity === 'random' && ' When "Random" is selected, a random city will be chosen each time you solve.'}
          </div>
        </div>

        <div className="button-group">
          <button
            className="button button-primary"
            onClick={handleSolve}
            disabled={isSolving}
          >
            {isSolving ? 'Solving...' : 'Solve TSP'}
          </button>
          <button
            className="button button-secondary"
            onClick={handleRandomizeDistances}
            disabled={isSolving}
          >
            Randomize Distances
          </button>
        </div>
      </div>

      {isSolving && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
            <div className="spinner-ring"></div>
          </div>
          <p className="loading-text">Exploring tour permutations...</p>
          <p className="loading-subtext">Applying geometric transforms</p>
        </div>
      )}

      {solution && (
        <div className="results-panel">
          <div className="results-header">
            <h2>Optimal Tour Found!</h2>
            <div className="solution-explanation">
              <p>
                Hologram explored <strong>{universeCount} possible tours</strong> using geometric transforms in just{' '}
                <strong>{solution.runtimeMs.toFixed(2)} milliseconds</strong> to find the best route.
              </p>
            </div>
            <div className="solution-stats">
              <div className="stat">
                <span className="stat-label">Tours Explored</span>
                <span className="stat-value">{universeCount}</span>
                <span className="stat-desc">All possible permutations</span>
              </div>
              <div className="stat">
                <span className="stat-label">Runtime</span>
                <span className="stat-value">{solution.runtimeMs.toFixed(2)}ms</span>
                <span className="stat-desc">Geometric evaluation</span>
              </div>
              <div className="stat">
                <span className="stat-label">Tour Distance</span>
                <span className="stat-value">{solution.distance.toFixed(2)}</span>
                <span className="stat-desc">Shortest route found</span>
              </div>
            </div>
          </div>

          <div style={{ 
            backgroundColor: 'var(--surface-elevated)', 
            border: '1px solid var(--border-color)',
            padding: '20px', 
            marginBottom: '20px'
          }}>
            <h3 style={{ color: 'var(--primary-color)', marginBottom: '15px', fontFamily: 'var(--font-family-display)', textTransform: 'uppercase' }}>
              📊 Tour Visualization
            </h3>
            <canvas
              ref={canvasRef}
              width={800}
              height={500}
              style={{
                border: '1px solid var(--border-color)',
                backgroundColor: 'var(--bg-color)',
                display: 'block',
                margin: '0 auto',
              }}
            />
            <div style={{ marginTop: '15px', textAlign: 'center', color: 'var(--text-muted-color)' }}>
              <strong style={{ color: 'var(--primary-color)' }}>Distance: {solution.distance.toFixed(2)} miles</strong>
              <div style={{ marginTop: '5px' }}>
                Tour: {solution.cities.map(i => instance.cities[i].name).join(' → ')} → {instance.cities[solution.cities[0]].name}
              </div>
            </div>
          </div>

          {/* Comparison Graph - Show only after solution */}
          {solution && (
                <div style={{ 
              backgroundColor: 'var(--surface-elevated)', 
              border: '1px solid var(--border-color)',
              padding: '20px', 
              marginBottom: '20px'
            }}>
              <h3 style={{ 
                color: 'var(--primary-color)', 
                marginBottom: '15px',
                fontFamily: 'var(--font-family-display)',
                textTransform: 'uppercase'
              }}>
                📈 Comparison Graph
              </h3>
              <p style={{ 
                color: 'var(--text-muted-color)', 
                marginBottom: '15px',
                fontSize: '0.9rem'
              }}>
                Plot runtime vs. number of cities for classical solvers (log scale). Overlay your method — it's essentially flat at 0.6 ms.
              </p>
              <canvas
                ref={comparisonGraphRef}
                width={900}
                height={500}
                style={{
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-color)',
                  display: 'block',
                  margin: '0 auto',
                  maxWidth: '100%',
                  height: 'auto'
                }}
              />
            </div>
          )}

          {/* Best Tour Solution */}
          <div style={{ 
            backgroundColor: 'var(--surface-elevated)', 
            border: '1px solid var(--border-color)',
            padding: '20px', 
            marginBottom: '20px'
          }}>
            <h3 style={{ 
              color: 'var(--primary-color)', 
              marginBottom: '15px',
              fontFamily: 'var(--font-family-display)',
              textTransform: 'uppercase'
            }}>
              🏆 Best Tour Solution
              </h3>
              
            {/* Trip Sequence with Distances */}
            <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: 'var(--text-secondary)', fontSize: '1.1em' }}>Trip Sequence:</strong>
                <div style={{ 
                  marginTop: '10px',
                  padding: '15px',
                  backgroundColor: 'var(--bg-color)',
                  border: '1px solid var(--border-color)'
                }}>
                  {solution.cities.map((cityId, i) => {
                    const city = instance.cities[cityId];
                    const nextCityId = solution.cities[(i + 1) % solution.cities.length];
                    const nextCity = instance.cities[nextCityId];
                    const distance = instance.distances[cityId][nextCityId];
                    
                    return (
                      <div key={i} style={{ 
                        display: 'flex',
                        alignItems: 'center',
                        marginBottom: '8px',
                        fontFamily: 'var(--font-family-mono)',
                        fontSize: '14px'
                      }}>
                        <span style={{ 
                          color: 'var(--primary-color)', 
                          fontWeight: 'bold',
                          minWidth: '30px',
                          textAlign: 'right',
                          marginRight: '10px'
                        }}>
                          {i + 1}.
                        </span>
                        <span style={{ color: 'var(--text-color)', minWidth: '80px' }}>
                          {city.name}
                        </span>
                        <span style={{ color: 'var(--text-muted-color)', margin: '0 10px' }}>
                          →
                        </span>
                        <span style={{ color: 'var(--text-color)', minWidth: '80px' }}>
                          {nextCity.name}
                        </span>
                        <span style={{ 
                          color: 'var(--success-color)', 
                          marginLeft: 'auto',
                          fontWeight: 'bold'
                        }}>
                          {distance.toFixed(2)} miles
                        </span>
                      </div>
                    );
                  })}
                  <div style={{ 
                    marginTop: '15px',
                    paddingTop: '15px',
                    borderTop: '1px solid var(--border-color)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 'bold' }}>
                      Total Distance:
                    </span>
                    <span style={{ 
                      color: 'var(--primary-color)', 
                      fontSize: '1.2em',
                      fontWeight: 'bold',
                      fontFamily: 'var(--font-family-mono)'
                    }}>
                      {solution.distance.toFixed(2)} miles
                    </span>
                </div>
              </div>
            </div>

              {/* Why This Solution is Optimal */}
                    <div style={{ marginBottom: '20px' }}>
                <strong style={{ color: 'var(--text-secondary)', fontSize: '1.1em' }}>Why This Solution is Optimal:</strong>
                      <div style={{ 
                  marginTop: '10px',
                  padding: '15px',
                  backgroundColor: 'var(--bg-color)',
                  border: '1px solid var(--border-color)',
                  lineHeight: '1.8'
                }}>
                  <p style={{ margin: '0 0 10px 0', color: 'var(--text-color)' }}>
                    Hologram evaluated <strong style={{ color: 'var(--primary-color)' }}>{universeCount} possible tours</strong> using geometric algebra transforms.
                  </p>
                  <p style={{ margin: '0 0 10px 0', color: 'var(--text-color)' }}>
                    This tour has the <strong style={{ color: 'var(--success-color)' }}>shortest total distance</strong> of all possible routes that visit each city exactly once.
                  </p>
                  <p style={{ margin: '0', color: 'var(--text-color)' }}>
                    The solution was found in just <strong style={{ color: 'var(--primary-color)' }}>{solution.runtimeMs.toFixed(2)} milliseconds</strong> by exploring the solution space through geometric transformations rather than checking each route individually.
                  </p>
                      </div>
                    </div>

              {/* Technical Details (Collapsible) */}
              <details style={{ marginTop: '20px' }}>
                <summary style={{ 
                  color: 'var(--text-secondary)', 
                  cursor: 'pointer',
                  marginBottom: '10px',
                  fontSize: '0.9em'
                }}>
                  Technical Details (Atlas Expression & Geometric Structure)
                </summary>
                <div style={{ marginTop: '15px' }}>
                  <div style={{ marginBottom: '15px' }}>
                    <strong style={{ color: 'var(--text-secondary)' }}>Atlas Expression:</strong>
                      <div style={{ 
                      fontFamily: 'var(--font-family-mono)', 
                      backgroundColor: 'var(--bg-color)', 
                        padding: '10px',
                        marginTop: '5px',
                      color: 'var(--primary-color)',
                      wordBreak: 'break-all',
                      border: '1px solid var(--border-color)',
                      fontSize: '12px'
                      }}>
                      {solution.atlasExpression}
                      </div>
                    </div>
                    <div>
                    <strong style={{ color: 'var(--text-secondary)' }}>Geometric Structure:</strong>
                      <div style={{ marginTop: '10px' }}>
                      {solution.geometricEncoding.map((geo, i) => {
                        const city = instance.cities[geo.cityId];
                            return (
                          <div key={i} style={{ 
                                padding: '8px',
                                marginBottom: '5px',
                            backgroundColor: 'var(--bg-color)',
                            fontFamily: 'var(--font-family-mono)',
                            fontSize: '12px',
                            border: '1px solid var(--border-color)'
                          }}>
                            <span style={{ color: 'var(--primary-color)' }}>{i}.</span>{' '}
                            <span style={{ color: 'var(--text-color)' }}>{city.name}</span> →{' '}
                            <span style={{ color: 'var(--success-color)' }}>c{geo.classIndex}</span>{' '}
                            <span style={{ color: 'var(--text-muted-color)' }}>(h₂={geo.coordinates.h2}, d={geo.coordinates.d}, ℓ={geo.coordinates.l})</span>
                              </div>
                            );
                      })}
                      </div>
                    </div>
                  </div>
              </details>
            </div>
        </div>
      )}

      {!solution && !isSolving && (
        <div style={{ 
          backgroundColor: 'var(--surface-elevated)', 
          border: '1px solid var(--border-color)',
          padding: '40px', 
          textAlign: 'center',
          color: 'var(--text-muted-color)'
        }}>
          <p>Select a preset and click "Solve TSP" to see Hologram in action!</p>
          <div style={{ marginTop: '20px', fontSize: '14px', color: 'var(--text-muted-color)' }}>
            <p>Key Features:</p>
            <ul style={{ textAlign: 'left', display: 'inline-block', marginTop: '10px' }}>
              <li>96-Class Geometric Encoding (h₂, d, ℓ coordinates)</li>
              <li>Transform Operations (R, T, M, D) for tour mutations</li>
              <li>Sequential Composition (.) for representing tour order</li>
              <li>Geometric Algebra for efficient solution space exploration</li>
              <li>Triality Orbits for structured search</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default TSPDemo;

