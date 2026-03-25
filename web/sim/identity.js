/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  Deterministic identity generator for simulation entities.
  JS port of Python build_identity() from tritium_lib.sim_engine.core.entity.

  All generation is deterministic per entity ID — the same entity always
  gets the same identity across sessions. Uses a simple hash-based seeded
  PRNG to mirror the Python implementation.
*/

// =========================================================================
// Seeded PRNG (Mulberry32)
// =========================================================================
function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function simpleHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

function seededRng(entityId, salt = '') {
  const seed = simpleHash(entityId + ':' + salt);
  return mulberry32(seed);
}

// =========================================================================
// West Dublin, CA data (mirroring Python entity.py)
// =========================================================================

const FIRST_NAMES = [
  'James', 'Maria', 'David', 'Priya', 'Carlos', 'Linda',
  'Aiden', 'Susan', 'Derek', 'Tommy', 'Patricia', 'Miguel',
  'Jennifer', 'Robert', 'Mei', 'Omar', 'Elena', 'Jamal',
  'Kenji', 'Sarah', 'Andre', 'Rosa', 'Kevin', 'Fatima',
  'Brian', 'Yuki', 'Marcus', 'Lila', 'Trevor', 'Anita',
  'Hassan', 'Christine', 'Raj', 'Donna', 'Tyrell', 'Nina',
];

const LAST_NAMES = [
  'Nakamura', 'Okafor', 'Venkatesh', 'Reyes', 'Medina',
  'Walsh', 'Park', 'Chen', 'Gutierrez', 'Patel', 'Kim',
  'Johnson', 'Williams', 'Santos', 'Garcia', 'Martinez',
  'Nguyen', 'Tanaka', "O'Brien", 'Murphy', 'Larsen',
  'Petrova', 'Ali', 'Singh', 'Moreira', 'Campbell',
  'Friedman', 'Kowalski', 'Adams', 'Torres',
];

const DUBLIN_STREETS = [
  'Dublin Blvd', 'Village Parkway', 'Silvergate Dr', 'Alegre Dr',
  'Hacienda Dr', 'Tassajara Rd', 'Gleason Dr', 'Amador Valley Blvd',
  'Dougherty Rd', 'Clark Ave', 'Briar Rose Ln', 'Emerald Glen Dr',
  'Iron Horse Pkwy', 'Regional St', 'Golden Gate Dr', 'Grafton St',
  'Sierra Ct', 'Canyon Creek Cir', 'Martindale Ct', 'Stagecoach Rd',
  'Donner Way', 'Donlon Way', 'Scarlett Dr', 'San Ramon Rd',
  'Antone Way', 'Central Pkwy', 'Lockhart St', 'Finnian Way',
  'Kolln St', 'Hansen Dr',
];

const DUBLIN_BUSINESSES = [
  'Valley Auto Repair', 'Dublin Ranch Dental', 'Tri-Valley Fitness',
  'Lucky Supermarket', 'Whole Foods Dublin', "Peet's Coffee",
  'Safeway Distribution', 'Kaiser Permanente Dublin', 'Target Dublin',
  'Ross Dress for Less', "Raley's", 'Starbucks Village Parkway',
  'Dublin Toyota', 'Bay Club', 'Round Table Pizza',
  'Wells Fargo Dublin Blvd', 'Chase Bank Hacienda', 'In-N-Out Burger',
  'Tractor Supply Co', 'The UPS Store Dublin', 'Subway Dougherty',
  'Emerald Glen Recreation', 'Dublin Sports Grounds', 'Habit Burger',
];

const PHONE_MODELS = [
  'iPhone 15 Pro', 'iPhone 15', 'iPhone 14 Pro', 'iPhone 14',
  'iPhone SE', 'Samsung Galaxy S24', 'Samsung Galaxy S23',
  'Samsung Galaxy A54', 'Google Pixel 8 Pro', 'Google Pixel 8',
  'Google Pixel 7a', 'Samsung Galaxy Z Flip5',
];

// OUI prefixes for realistic MAC generation
const APPLE_OUIS = ['A4:C3:F0', '3C:E0:72', '78:7B:8A', '9C:20:7B', 'F0:18:98'];
const SAMSUNG_OUIS = ['8C:F5:A3', 'C0:BD:D1', '50:01:D9', 'AC:5F:3E', 'B4:3A:28'];
const GOOGLE_OUIS = ['54:60:09', '94:EB:2C', 'F8:8F:CA'];

const VEHICLE_MAKES = [
  { make: 'Toyota', models: ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Tacoma', 'Prius'] },
  { make: 'Honda', models: ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey'] },
  { make: 'Ford', models: ['F-150', 'Escape', 'Explorer', 'Mustang', 'Maverick'] },
  { make: 'Chevrolet', models: ['Silverado', 'Malibu', 'Equinox', 'Tahoe', 'Bolt EV'] },
  { make: 'Tesla', models: ['Model 3', 'Model Y', 'Model S', 'Model X'] },
  { make: 'Subaru', models: ['Outback', 'Forester', 'Crosstrek', 'Impreza'] },
  { make: 'BMW', models: ['3 Series', 'X3', 'X5', '5 Series'] },
  { make: 'Hyundai', models: ['Elantra', 'Tucson', 'Santa Fe', 'Ioniq 5'] },
  { make: 'Nissan', models: ['Altima', 'Rogue', 'Sentra', 'Leaf'] },
  { make: 'Kia', models: ['Sorento', 'Sportage', 'Forte', 'EV6'] },
];

const VEHICLE_COLORS = [
  'White', 'Black', 'Silver', 'Gray', 'Blue', 'Red',
  'Green', 'Beige', 'Dark Blue', 'Maroon', 'Gold', 'Brown',
];

// =========================================================================
// Generator functions
// =========================================================================

function pick(rng, arr) {
  return arr[Math.floor(rng() * arr.length)];
}

function generateMac(rng, device = 'wifi') {
  // Pick an OUI prefix based on a random phone type
  const phoneRoll = rng();
  let ouis;
  if (phoneRoll < 0.45) ouis = APPLE_OUIS;
  else if (phoneRoll < 0.75) ouis = SAMSUNG_OUIS;
  else ouis = GOOGLE_OUIS;

  const prefix = pick(rng, ouis);
  const suffix = [
    Math.floor(rng() * 256),
    Math.floor(rng() * 256),
    Math.floor(rng() * 256),
  ].map(b => b.toString(16).toUpperCase().padStart(2, '0')).join(':');
  return prefix + ':' + suffix;
}

function generatePlate(rng) {
  const d1 = 1 + Math.floor(rng() * 9);
  const letters = 'ABCDEFGHJKLMNPRSTUVWXYZ';
  const l1 = letters[Math.floor(rng() * letters.length)];
  const l2 = letters[Math.floor(rng() * letters.length)];
  const l3 = letters[Math.floor(rng() * letters.length)];
  const d3 = 100 + Math.floor(rng() * 900);
  return '' + d1 + l1 + l2 + l3 + d3;
}

function generateAddress(rng) {
  const number = 1000 + Math.floor(rng() * 9000);
  return number + ' ' + pick(rng, DUBLIN_STREETS);
}

// =========================================================================
// Main identity builder
// =========================================================================

/**
 * Build a deterministic identity for a simulation entity.
 * @param {string} entityId - Unique entity identifier (e.g. 'ped_0', 'car_3')
 * @param {string} entityType - 'person' or 'vehicle'
 * @returns {object} Identity object with all relevant fields
 */
export function buildIdentity(entityId, entityType = 'person') {
  const rng = seededRng(entityId, 'identity');

  const firstName = pick(rng, FIRST_NAMES);
  const lastName = pick(rng, LAST_NAMES);

  const identity = {
    shortId: simpleHash(entityId).toString(16).slice(-6).toUpperCase().padStart(6, '0'),
    firstName,
    lastName,
    fullName: firstName + ' ' + lastName,
    bluetoothMac: generateMac(rng, 'bluetooth'),
    wifiMac: generateMac(rng, 'wifi'),
    phoneModel: pick(rng, PHONE_MODELS),
    employer: pick(rng, DUBLIN_BUSINESSES),
    homeAddress: generateAddress(rng),
    workAddress: generateAddress(rng),
  };

  if (entityType === 'vehicle') {
    const makeEntry = pick(rng, VEHICLE_MAKES);
    identity.licensePlate = generatePlate(rng);
    identity.vehicleMake = makeEntry.make;
    identity.vehicleModel = pick(rng, makeEntry.models);
    identity.vehicleYear = 2018 + Math.floor(rng() * 8);
    identity.vehicleColor = pick(rng, VEHICLE_COLORS);
    identity.vehicleDesc = identity.vehicleYear + ' ' + identity.vehicleMake + ' ' + identity.vehicleModel;
    // Owner is the person identity
    identity.ownerName = firstName + ' ' + lastName;
  }

  return identity;
}

/**
 * Convenience: build identity for a pedestrian by index.
 * @param {number} index
 * @returns {object}
 */
export function buildPedestrianIdentity(index) {
  return buildIdentity('ped_' + index, 'person');
}

/**
 * Convenience: build identity for a car by index.
 * @param {number} index
 * @returns {object}
 */
export function buildCarIdentity(index) {
  return buildIdentity('car_' + index, 'vehicle');
}
