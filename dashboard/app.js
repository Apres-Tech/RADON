/**
 * RADON — Autonomous Multi-Agent Rescue Simulation 3D Dashboard
 * Interactive Three.js multi-floor viewer with live replay playback,
 * held-out building generator, and adversarial fire visualization.
 */

// Configuration & Global State
const CONFIG = {
  floorHeight: 3.6,
  cellSize: 1.0,
  wallHeight: 1.0,
  agentColors: [
    0x06b6d4, // Rescuer 0: Cyan
    0x3b82f6, // Rescuer 1: Blue
    0xa855f7, // Rescuer 2: Purple
    0xf59e0b, // Rescuer 3: Amber
  ],
  hostageColor: 0x10b981, // Emerald Green
  exitColor: 0x22c55e,    // Bright Green
  stairsColor: 0xeab308,  // Gold / Yellow
  fireColor: 0xef4444,    // Crimson Fire
  fireGlowColor: 0xff6b00 // Bright Orange Glow
};

let scene, camera, renderer, controls;
let currentReplay = null;
let currentFrameIndex = 0;
let isPlaying = false;
let playbackSpeed = 1.0;
let playbackInterval = null;
let currentFloorFilter = "both"; // "both", "0", "1"
let currentFireMode = "curriculum";

// Three.js object groups
let buildingGroup = null;
let entityGroup = null;
let floorPlanes = [];
let floorMeshes = { 0: [], 1: [] };

// Dynamic entity meshes cache
let rescuerMeshes = {};
let hostageMeshes = [];
let fireMeshes = [];

// DOM Elements
const dom = {
  canvasContainer: document.getElementById("canvas-container"),
  backendStatusPill: document.getElementById("backend-status-pill"),
  backendStatusText: document.getElementById("backend-status-text"),
  statusDot: document.getElementById("status-dot"),
  devicePill: document.getElementById("device-pill"),
  deviceText: document.getElementById("device-text"),
  replaySelect: document.getElementById("replay-select"),
  fireModeControl: document.getElementById("fire-mode-control"),
  fireModeCaption: document.getElementById("fire-mode-caption"),
  generateBtn: document.getElementById("generate-btn"),
  generateSpinner: document.getElementById("generate-spinner"),
  generateIcon: document.getElementById("generate-icon"),
  generateBtnText: document.getElementById("generate-btn-text"),
  playPauseBtn: document.getElementById("play-pause-btn"),
  playIcon: document.getElementById("play-icon"),
  pauseIcon: document.getElementById("pause-icon"),
  prevFrameBtn: document.getElementById("prev-frame-btn"),
  nextFrameBtn: document.getElementById("next-frame-btn"),
  timelineSlider: document.getElementById("timeline-slider"),
  currentFrameLbl: document.getElementById("current-frame-lbl"),
  totalFramesLbl: document.getElementById("total-frames-lbl"),
  completionTag: document.getElementById("episode-completion-tag"),
  statRescued: document.getElementById("stat-rescued"),
  statSurvivors: document.getElementById("stat-survivors"),
  statFire: document.getElementById("stat-fire"),
  statStep: document.getElementById("stat-step"),
  rescuerList: document.getElementById("rescuer-list"),
  resetCamBtn: document.getElementById("reset-cam-btn"),
  openAnalyticsBtn: document.getElementById("open-analytics-btn"),
  analyticsModal: document.getElementById("analytics-modal"),
  closeModalBtn: document.getElementById("close-modal-btn"),
  toast: document.getElementById("toast-notification"),
  toastMessage: document.getElementById("toast-message"),
  metaCard: document.getElementById("active-episode-meta"),
  metaName: document.getElementById("meta-name"),
  metaSeed: document.getElementById("meta-seed"),
  metaOutcome: document.getElementById("meta-outcome"),
};

// Initialize 3D Engine
function initThree() {
  const width = dom.canvasContainer.clientWidth || window.innerWidth;
  const height = dom.canvasContainer.clientHeight || window.innerHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  scene.fog = new THREE.FogExp2(0x0a0e17, 0.015);

  // Camera
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(16, 22, 22);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  dom.canvasContainer.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.target.set(5, 2, 5);

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xfff8ee, 0.85);
  mainLight.position.set(18, 28, 14);
  mainLight.castShadow = true;
  mainLight.shadow.mapSize.width = 1024;
  mainLight.shadow.mapSize.height = 1024;
  mainLight.shadow.camera.near = 0.5;
  mainLight.shadow.camera.far = 80;
  mainLight.shadow.camera.left = -20;
  mainLight.shadow.camera.right = 20;
  mainLight.shadow.camera.top = 20;
  mainLight.shadow.camera.bottom = -20;
  scene.add(mainLight);

  const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
  fillLight.position.set(-15, 15, -15);
  scene.add(fillLight);

  // Groups
  buildingGroup = new THREE.Group();
  entityGroup = new THREE.Group();
  scene.add(buildingGroup);
  scene.add(entityGroup);

  // Resize handler
  window.addEventListener("resize", onWindowResize);

  // Render loop
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Subtle pulsing animation on fire meshes
  const time = performance.now() * 0.005;
  fireMeshes.forEach((mesh, idx) => {
    if (mesh.visible) {
      const pulse = 1.0 + 0.15 * Math.sin(time + idx);
      mesh.scale.set(pulse, pulse, pulse);
    }
  });

  renderer.render(scene, camera);
}

function onWindowResize() {
  const width = dom.canvasContainer.clientWidth || window.innerWidth;
  const height = dom.canvasContainer.clientHeight || window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// Reset camera to default isometric view
function resetCamera() {
  if (!currentReplay) {
    camera.position.set(16, 22, 22);
    controls.target.set(5, 2, 5);
  } else {
    const w = currentReplay.config.grid_w || 10;
    const h = currentReplay.config.grid_h || 10;
    const cx = (w - 1) / 2.0;
    const cz = (h - 1) / 2.0;
    camera.position.set(cx + 12, 18, cz + 16);
    controls.target.set(cx, 1.8, cz);
  }
  controls.update();
}

// Show Toast Notification
function showToast(message, type = "info") {
  dom.toastMessage.textContent = message;
  dom.toast.className = `toast-notification ${type}`;
  dom.toast.style.display = "flex";
  setTimeout(() => {
    dom.toast.style.display = "none";
  }, 4500);
}

// Build 3D Building Geometry from Replay
function buildBuilding(replayData) {
  // Clear existing building
  while (buildingGroup.children.length > 0) {
    const obj = buildingGroup.children[0];
    buildingGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  floorMeshes = { 0: [], 1: [] };
  floorPlanes = [];

  const config = replayData.config;
  const mapData = replayData.map;
  const floors = config.floors || 2;
  const gridH = config.grid_h || 10;
  const gridW = config.grid_w || 10;

  // Center target
  controls.target.set((gridW - 1) / 2.0, (floors * CONFIG.floorHeight) / 4.0, (gridH - 1) / 2.0);

  // Materials
  const floorMaterial0 = new THREE.MeshStandardMaterial({
    color: 0x141b26,
    roughness: 0.8,
    metalness: 0.2,
  });

  const floorMaterial1 = new THREE.MeshStandardMaterial({
    color: 0x1a2638,
    roughness: 0.6,
    metalness: 0.3,
    transparent: true,
    opacity: 0.75,
  });

  const wallMaterial = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.7,
    metalness: 0.4,
  });

  const wallEdgeMaterial = new THREE.LineBasicMaterial({ color: 0x64748b, linewidth: 1 });

  const exitMaterial = new THREE.MeshStandardMaterial({
    color: CONFIG.exitColor,
    emissive: 0x15803d,
    emissiveIntensity: 0.7,
    roughness: 0.3,
  });

  const stairsMaterial = new THREE.MeshStandardMaterial({
    color: CONFIG.stairsColor,
    emissive: 0xb45309,
    emissiveIntensity: 0.6,
    roughness: 0.4,
  });

  // 1. Build Floor Slabs & Grids
  for (let f = 0; f < floors; f++) {
    const yPos = f * CONFIG.floorHeight;

    // Floor plane geometry
    const planeGeo = new THREE.BoxGeometry(gridW * CONFIG.cellSize, 0.1, gridH * CONFIG.cellSize);
    const planeMat = f === 0 ? floorMaterial0 : floorMaterial1;
    const planeMesh = new THREE.Mesh(planeGeo, planeMat);
    planeMesh.position.set((gridW - 1) / 2.0, yPos - 0.05, (gridH - 1) / 2.0);
    planeMesh.receiveShadow = true;
    buildingGroup.add(planeMesh);
    floorPlanes.push(planeMesh);
    floorMeshes[f].push(planeMesh);

    // Floor grid lines
    const gridHelper = new THREE.GridHelper(gridW, gridW, 0x475569, 0x1e293b);
    gridHelper.position.set((gridW - 1) / 2.0, yPos + 0.01, (gridH - 1) / 2.0);
    buildingGroup.add(gridHelper);
    floorMeshes[f].push(gridHelper);

    // Floor label badge in 3D
    const floorTagGeo = new THREE.BoxGeometry(1.2, 0.02, 0.4);
    const floorTagMat = new THREE.MeshBasicMaterial({ color: f === 0 ? 0x10b981 : 0x3b82f6 });
    const floorTag = new THREE.Mesh(floorTagGeo, floorTagMat);
    floorTag.position.set(-1.0, yPos + 0.02, 0);
    buildingGroup.add(floorTag);
    floorMeshes[f].push(floorTag);
  }

  // 2. Build Walls, Exits, and Stairs from static map layout
  const mapFloors = mapData.floors || [];
  for (let f = 0; f < mapFloors.length; f++) {
    const grid = mapFloors[f];
    const yBase = f * CONFIG.floorHeight;

    for (let r = 0; r < gridH; r++) {
      for (let c = 0; c < gridW; c++) {
        const tileVal = grid[r][c];

        // WALL (Tile ID 1)
        if (tileVal === 1) {
          const wallGeo = new THREE.BoxGeometry(0.94, CONFIG.wallHeight, 0.94);
          const wallMesh = new THREE.Mesh(wallGeo, wallMaterial);
          wallMesh.position.set(c, yBase + CONFIG.wallHeight / 2.0, r);
          wallMesh.castShadow = true;
          wallMesh.receiveShadow = true;

          // Subtle wireframe edge
          const edges = new THREE.EdgesGeometry(wallGeo);
          const line = new THREE.LineSegments(edges, wallEdgeMaterial);
          wallMesh.add(line);

          buildingGroup.add(wallMesh);
          floorMeshes[f].push(wallMesh);
        }
      }
    }
  }

  // 3. Exits (marked on Floor 0)
  const exits = mapData.exits || [];
  exits.forEach(([ef, er, ec]) => {
    const exitPadGeo = new THREE.BoxGeometry(0.92, 0.06, 0.92);
    const exitMesh = new THREE.Mesh(exitPadGeo, exitMaterial);
    exitMesh.position.set(ec, 0.03, er);
    buildingGroup.add(exitMesh);
    floorMeshes[0].push(exitMesh);

    // Exit beacon cylinder
    const beaconGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.7, 12);
    const beaconMat = new THREE.MeshBasicMaterial({ color: 0x4ade80, transparent: true, opacity: 0.6 });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.set(ec, 0.38, er);
    buildingGroup.add(beacon);
    floorMeshes[0].push(beacon);
  });

  // 4. Stairs (connecting Floor 0 and Floor 1)
  const stairs = mapData.stairs || [];
  stairs.forEach(([sr, sc]) => {
    // Ground stair pad
    const stairPadGeo = new THREE.BoxGeometry(0.9, 0.08, 0.9);
    const stair0 = new THREE.Mesh(stairPadGeo, stairsMaterial);
    stair0.position.set(sc, 0.04, sr);
    buildingGroup.add(stair0);
    floorMeshes[0].push(stair0);

    // Upper stair pad
    if (floors > 1) {
      const stair1 = new THREE.Mesh(stairPadGeo, stairsMaterial);
      stair1.position.set(sc, CONFIG.floorHeight + 0.04, sr);
      buildingGroup.add(stair1);
      floorMeshes[1].push(stair1);

      // Connecting vertical stair shaft / spiral column
      const shaftGeo = new THREE.CylinderGeometry(0.2, 0.2, CONFIG.floorHeight, 8);
      const shaftMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.8, roughness: 0.2 });
      const shaft = new THREE.Mesh(shaftGeo, shaftMat);
      shaft.position.set(sc, CONFIG.floorHeight / 2.0, sr);
      buildingGroup.add(shaft);
      floorMeshes[0].push(shaft);
      floorMeshes[1].push(shaft);

      // Amber stairs beacon light
      const stairLight = new THREE.PointLight(0xf59e0b, 0.5, 4.0);
      stairLight.position.set(sc, CONFIG.floorHeight / 2.0, sr);
      buildingGroup.add(stairLight);
    }
  });

  applyFloorFilter(currentFloorFilter);
}

// Apply floor filter (both, floor 0, floor 1)
function applyFloorFilter(filter) {
  currentFloorFilter = filter;
  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.floor === filter);
  });

  if (filter === "both") {
    if (floorMeshes[0]) floorMeshes[0].forEach(m => m.visible = true);
    if (floorMeshes[1]) floorMeshes[1].forEach(m => m.visible = true);
  } else if (filter === "0") {
    if (floorMeshes[0]) floorMeshes[0].forEach(m => m.visible = true);
    if (floorMeshes[1]) floorMeshes[1].forEach(m => m.visible = false);
  } else if (filter === "1") {
    if (floorMeshes[0]) floorMeshes[0].forEach(m => m.visible = false);
    if (floorMeshes[1]) floorMeshes[1].forEach(m => m.visible = true);
  }

  // Update entity visibility
  updateEntityVisibility();
}

function updateEntityVisibility() {
  if (!currentReplay || !currentReplay.frames || !currentReplay.frames[currentFrameIndex]) return;
  const frame = currentReplay.frames[currentFrameIndex];

  // Rescuers
  frame.rescuers.forEach(r => {
    const mesh = rescuerMeshes[r.id];
    if (mesh) {
      if (currentFloorFilter === "both") mesh.visible = true;
      else mesh.visible = (String(r.floor) === currentFloorFilter);
    }
  });

  // Hostages
  frame.hostages.forEach((h, idx) => {
    const mesh = hostageMeshes[idx];
    if (mesh) {
      const isRescued = h.length > 3 && h[3];
      if (isRescued) {
        mesh.visible = false;
      } else if (currentFloorFilter === "both") {
        mesh.visible = true;
      } else {
        mesh.visible = (String(h[0]) === currentFloorFilter);
      }
    }
  });

  // Fire
  frame.fire.forEach((fp, idx) => {
    const mesh = fireMeshes[idx];
    if (mesh) {
      if (currentFloorFilter === "both") mesh.visible = true;
      else mesh.visible = (String(fp[0]) === currentFloorFilter);
    }
  });
}

// Build Entity Meshes Pool
function initEntityPool(config) {
  // Clear entity group
  while (entityGroup.children.length > 0) {
    const obj = entityGroup.children[0];
    entityGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  rescuerMeshes = {};
  hostageMeshes = [];
  fireMeshes = [];

  const nRescuers = config.n_rescuers || 3;
  const nHostages = config.n_hostages || 2;
  const maxFires = 64; // Pool of fire meshes

  // 1. Rescuer Meshes (Capsule body + head + carrying pack)
  for (let i = 0; i < nRescuers; i++) {
    const rescuerId = `rescuer_${i}`;
    const agentColor = CONFIG.agentColors[i % CONFIG.agentColors.length];

    const group = new THREE.Group();

    // Body
    const bodyGeo = new THREE.CylinderGeometry(0.24, 0.28, 0.65, 12);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: agentColor,
      roughness: 0.4,
      metalness: 0.3,
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.38;
    body.castShadow = true;
    group.add(body);

    // Head / Helmet
    const headGeo = new THREE.SphereGeometry(0.18, 12, 12);
    const headMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 0.78;
    group.add(head);

    // Visor light
    const visorGeo = new THREE.BoxGeometry(0.2, 0.08, 0.12);
    const visorMat = new THREE.MeshBasicMaterial({ color: agentColor });
    const visor = new THREE.Mesh(visorGeo, visorMat);
    visor.position.set(0, 0.78, 0.15);
    group.add(visor);

    // Carrying backpack indicator (hidden by default)
    const packGeo = new THREE.BoxGeometry(0.3, 0.3, 0.22);
    const packMat = new THREE.MeshStandardMaterial({ color: CONFIG.hostageColor, emissive: 0x059669, emissiveIntensity: 0.6 });
    const pack = new THREE.Mesh(packGeo, packMat);
    pack.position.set(0, 0.48, -0.22);
    pack.visible = false;
    group.add(pack);
    group.userData.pack = pack;

    // Fallen halo / skull hazard for dead agent
    const haloGeo = new THREE.RingGeometry(0.25, 0.32, 16);
    const haloMat = new THREE.MeshBasicMaterial({ color: 0xef4444, side: THREE.DoubleSide });
    const halo = new THREE.Mesh(haloGeo, haloMat);
    halo.rotation.x = Math.PI / 2;
    halo.position.y = 0.05;
    halo.visible = false;
    group.add(halo);
    group.userData.halo = halo;

    entityGroup.add(group);
    rescuerMeshes[rescuerId] = group;
  }

  // 2. Hostage Meshes (Green cylinder with medical cross aura)
  for (let i = 0; i < nHostages; i++) {
    const group = new THREE.Group();

    // Body
    const bodyGeo = new THREE.CylinderGeometry(0.2, 0.22, 0.55, 12);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: CONFIG.hostageColor,
      roughness: 0.3,
      emissive: 0x047857,
      emissiveIntensity: 0.4,
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.32;
    body.castShadow = true;
    group.add(body);

    // Head
    const headGeo = new THREE.SphereGeometry(0.15, 12, 12);
    const headMat = new THREE.MeshStandardMaterial({ color: 0xa7f3d0 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 0.68;
    group.add(head);

    // Medical aura ring
    const ringGeo = new THREE.RingGeometry(0.32, 0.38, 16);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x34d399, side: THREE.DoubleSide, transparent: true, opacity: 0.7 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.04;
    group.add(ring);

    entityGroup.add(group);
    hostageMeshes.push(group);
  }

  // 3. Fire Meshes Pool
  const fireGeo = new THREE.DodecahedronGeometry(0.38, 0);
  for (let i = 0; i < maxFires; i++) {
    const fireMat = new THREE.MeshStandardMaterial({
      color: CONFIG.fireColor,
      emissive: CONFIG.fireGlowColor,
      emissiveIntensity: 0.9,
      roughness: 0.2,
      wireframe: false,
    });
    const fire = new THREE.Mesh(fireGeo, fireMat);
    fire.visible = false;
    entityGroup.add(fire);
    fireMeshes.push(fire);
  }
}

// Update 3D Entities and Telemetry for Given Frame
function renderFrame(frameIdx) {
  if (!currentReplay || !currentReplay.frames || frameIdx >= currentReplay.frames.length) return;
  currentFrameIndex = frameIdx;
  const frame = currentReplay.frames[frameIdx];
  const config = currentReplay.config;

  // 1. Update Rescuers
  frame.rescuers.forEach(r => {
    const mesh = rescuerMeshes[r.id];
    if (mesh) {
      const yBase = r.floor * CONFIG.floorHeight;
      mesh.position.set(r.col, yBase, r.row);

      // Status
      if (!r.alive) {
        mesh.rotation.z = Math.PI / 2.2; // Tilted fallen
        mesh.position.y = yBase + 0.1;
        mesh.userData.halo.visible = true;
        mesh.userData.pack.visible = false;
      } else {
        mesh.rotation.z = 0;
        mesh.userData.halo.visible = false;
        mesh.userData.pack.visible = !!r.carrying;
      }
    }
  });

  // 2. Update Hostages
  frame.hostages.forEach((h, idx) => {
    const mesh = hostageMeshes[idx];
    if (mesh) {
      const floor = h[0];
      const row = h[1];
      const col = h[2];
      const rescued = h.length > 3 ? h[3] : false;

      if (rescued) {
        mesh.visible = false;
      } else {
        const yBase = floor * CONFIG.floorHeight;
        mesh.position.set(col, yBase, row);
        mesh.visible = (currentFloorFilter === "both" || String(floor) === currentFloorFilter);
      }
    }
  });

  // 3. Update Fire
  fireMeshes.forEach((mesh, idx) => {
    if (idx < frame.fire.length) {
      const fp = frame.fire[idx];
      const floor = fp[0];
      const row = fp[1];
      const col = fp[2];
      const yBase = floor * CONFIG.floorHeight;
      mesh.position.set(col, yBase + 0.42, row);
      mesh.visible = (currentFloorFilter === "both" || String(floor) === currentFloorFilter);
    } else {
      mesh.visible = false;
    }
  });

  // 4. Update HUD Telemetry
  const rescuedCount = frame.hostages.filter(h => h.length > 3 && h[3]).length;
  const survivingCount = frame.rescuers.filter(r => r.alive).length;
  const fireCount = frame.fire.length;

  dom.statRescued.innerHTML = `${rescuedCount} <span class="stat-unit">/ ${config.n_hostages}</span>`;
  dom.statSurvivors.innerHTML = `${survivingCount} <span class="stat-unit">/ ${config.n_rescuers}</span>`;
  dom.statFire.textContent = fireCount;
  dom.statStep.innerHTML = `${frame.t} <span class="stat-unit">/ ${config.max_steps}</span>`;

  // Rescuer cards
  let rescuerHtml = "";
  frame.rescuers.forEach((r, i) => {
    const colorHex = `#${CONFIG.agentColors[i % CONFIG.agentColors.length].toString(16).padStart(6, "0")}`;
    let badge = "";
    if (!r.alive) badge = '<span class="rescuer-status-badge badge-dead">Dead</span>';
    else if (r.carrying) badge = '<span class="rescuer-status-badge badge-carrying">Carrying Hostage</span>';
    else badge = '<span class="rescuer-status-badge badge-free">Free</span>';

    rescuerHtml += `
      <div class="rescuer-card">
        <div class="rescuer-id">
          <span class="agent-dot" style="background:${colorHex};"></span>
          <span>${r.id}</span>
        </div>
        <div>F${r.floor} (${r.row}, ${r.col})</div>
        <div>${badge}</div>
      </div>
    `;
  });
  dom.rescuerList.innerHTML = rescuerHtml;

  // Timeline UI
  dom.timelineSlider.value = frameIdx;
  dom.currentFrameLbl.textContent = frameIdx;

  // Status tag
  if (rescuedCount === config.n_hostages) {
    dom.completionTag.textContent = "All Evacuated (Success)";
    dom.completionTag.style.color = "#10b981";
  } else if (survivingCount === 0) {
    dom.completionTag.textContent = "Team Overrun (All Fallen)";
    dom.completionTag.style.color = "#ef4444";
  } else if (frameIdx === currentReplay.frames.length - 1) {
    dom.completionTag.textContent = `Completed (${rescuedCount}/${config.n_hostages} Evacuated)`;
    dom.completionTag.style.color = rescuedCount > 0 ? "#f59e0b" : "#94a3b8";
  } else {
    dom.completionTag.textContent = "In Progress";
    dom.completionTag.style.color = "#38bdf8";
  }
}

// Load and Display a Replay
function loadReplay(replayData, replayTitle = "Replay") {
  currentReplay = replayData;
  pausePlayback();

  // Rebuild building geometry
  buildBuilding(replayData);

  // Initialize entity pool
  initEntityPool(replayData.config);

  // Set timeline bounds
  const frameCount = replayData.frames.length;
  dom.timelineSlider.max = Math.max(0, frameCount - 1);
  dom.timelineSlider.value = 0;
  dom.totalFramesLbl.textContent = Math.max(0, frameCount - 1);

  // Metadata card
  dom.metaCard.style.display = "block";
  dom.metaName.textContent = replayTitle;
  dom.metaSeed.textContent = replayData.config.seed !== null ? replayData.config.seed : "Procedural (Random)";

  const lastFrame = replayData.frames[frameCount - 1];
  if (lastFrame) {
    const rescued = lastFrame.hostages.filter(h => h.length > 3 && h[3]).length;
    const all = rescued === replayData.config.n_hostages;
    dom.metaOutcome.textContent = `${rescued}/${replayData.config.n_hostages} Evacuated (${all ? 'Full Clear' : 'Partial'})`;
  }

  // Render initial frame
  renderFrame(0);
  resetCamera();

  // Auto-play
  startPlayback();
}

// Playback Control Functions
function startPlayback() {
  if (isPlaying) return;
  isPlaying = true;
  dom.playIcon.style.display = "none";
  dom.pauseIcon.style.display = "block";

  const delayMs = Math.round(120 / playbackSpeed);
  playbackInterval = setInterval(() => {
    if (!currentReplay || !currentReplay.frames) return;
    if (currentFrameIndex < currentReplay.frames.length - 1) {
      renderFrame(currentFrameIndex + 1);
    } else {
      // Loop or pause
      renderFrame(0);
    }
  }, delayMs);
}

function pausePlayback() {
  isPlaying = false;
  dom.playIcon.style.display = "block";
  dom.pauseIcon.style.display = "none";
  if (playbackInterval) {
    clearInterval(playbackInterval);
    playbackInterval = null;
  }
}

function togglePlayback() {
  if (isPlaying) pausePlayback();
  else startPlayback();
}

function setSpeed(speed) {
  playbackSpeed = speed;
  document.querySelectorAll(".speed-chip").forEach(chip => {
    chip.classList.toggle("active", parseFloat(chip.dataset.speed) === speed);
  });
  if (isPlaying) {
    pausePlayback();
    startPlayback();
  }
}

// Fetch Backend Status and Replay List
async function fetchBackendData() {
  // 1. Health check
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const data = await res.json();
      dom.statusDot.classList.remove("offline");
      dom.backendStatusText.textContent = data.rescuer_loaded ? "Backend Online (Model Ready)" : "Backend Online (No Model)";
      if (data.device) {
        dom.devicePill.style.display = "inline-flex";
        dom.deviceText.textContent = data.device.toUpperCase();
      }
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (e) {
    dom.statusDot.classList.add("offline");
    dom.backendStatusText.textContent = "Backend Offline (Viewing Mode)";
    dom.devicePill.style.display = "none";
  }

  // 2. Fetch replays
  try {
    const res = await fetch("/replays");
    if (res.ok) {
      const replays = await res.json();
      dom.replaySelect.innerHTML = "";

      if (replays.length === 0) {
        dom.replaySelect.innerHTML = '<option value="">No replays found in replays/</option>';
        return;
      }

      replays.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r.name;
        const rescuedTag = r.hostages_rescued !== undefined ? ` [${r.hostages_rescued}/${r.n_hostages} Rescued]` : "";
        opt.textContent = `${r.name} (${r.frames_count || 0} frames)${rescuedTag}`;
        dom.replaySelect.appendChild(opt);
      });

      // Default load first replay
      const defaultReplay = replays.find(r => r.name === "day5_best_heldout.json") ||
                            replays.find(r => r.name === "day5_trained.json") ||
                            replays[0];
      if (defaultReplay) {
        dom.replaySelect.value = defaultReplay.name;
        loadReplayFromServer(defaultReplay.name);
      }
    }
  } catch (e) {
    console.warn("Could not fetch replay list:", e);
    // If backend is down, load default fallback
    loadFallbackReplay();
  }
}

// Load a specific replay from backend server
async function loadReplayFromServer(replayName) {
  try {
    const res = await fetch(`/replay/${encodeURIComponent(replayName)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    loadReplay(data, replayName);
  } catch (e) {
    showToast(`Failed to load replay ${replayName}: ${e.message}`, "error");
  }
}

// Live Episode Generation on Unseen Map
async function generateUnseenEpisode() {
  dom.generateBtn.disabled = true;
  dom.generateSpinner.style.display = "inline-block";
  dom.generateIcon.style.display = "none";
  dom.generateBtnText.textContent = "Simulating Unseen Map...";

  try {
    const payload = {
      fire_mode: currentFireMode,
      seed: Math.floor(50000 + Math.random() * 49999),
      preset: "training_day4_base",
      max_steps: 160
    };

    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || "Server error");
    }

    const replayData = await res.json();
    const summary = replayData.summary || {};
    const title = `Live Unseen (Seed ${summary.seed}, ${summary.fire_mode})`;

    loadReplay(replayData, title);

    const outcomeText = summary.all_rescued
      ? `🎉 SUCCESS! All ${summary.hostages_rescued}/${summary.total_hostages} hostages evacuated in ${summary.steps} steps!`
      : `Episode completed: ${summary.hostages_rescued}/${summary.total_hostages} evacuated (${summary.rescuers_surviving}/${summary.total_rescuers} rescuers survived).`;

    showToast(`Seed ${summary.seed} generated! ${outcomeText}`, summary.all_rescued ? "success" : "info");

  } catch (e) {
    console.error("Live generation failed:", e);
    showToast(`Generation failed: ${e.message}. Using current replay.`, "error");
  } finally {
    dom.generateBtn.disabled = false;
    dom.generateSpinner.style.display = "none";
    dom.generateIcon.style.display = "inline-block";
    dom.generateBtnText.textContent = "Generate on Unseen Map";
  }
}

// Fallback synthetic replay if server unavailable
function loadFallbackReplay() {
  const fallback = {
    config: { floors: 2, grid_h: 10, grid_w: 10, n_rescuers: 3, n_hostages: 2, max_steps: 60, seed: 50001 },
    map: {
      floors: [
        [
          [1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,0,1,1,0,0,1,1,0,1],
          [1,0,1,6,0,0,1,0,0,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,0,1,0,0,0,0,1,0,1],
          [1,0,1,1,0,0,1,1,0,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,5,0,0,0,0,0,0,5,1],
          [1,1,1,1,1,1,1,1,1,1]
        ],
        [
          [1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,1,1,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,0,0,6,0,0,0,0,0,1],
          [1,1,1,0,0,0,0,1,1,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,1,1,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,1],
          [1,1,1,1,1,1,1,1,1,1]
        ]
      ],
      exits: [[0, 8, 1], [0, 8, 8]],
      stairs: [[3, 3]]
    },
    frames: [
      {
        t: 0,
        fire: [[0, 2, 7], [1, 2, 2]],
        hostages: [[0, 6, 4, false], [1, 7, 7, false]],
        rescuers: [
          { id: "rescuer_0", floor: 0, row: 8, col: 2, carrying: false, alive: true },
          { id: "rescuer_1", floor: 0, row: 8, col: 7, carrying: false, alive: true },
          { id: "rescuer_2", floor: 0, row: 8, col: 3, carrying: false, alive: true },
        ],
        rewards: { rescuer_0: 0, rescuer_1: 0, rescuer_2: 0 }
      },
      {
        t: 10,
        fire: [[0, 2, 7], [0, 2, 8], [1, 2, 2], [1, 2, 3]],
        hostages: [[0, 6, 4, true], [1, 7, 7, false]],
        rescuers: [
          { id: "rescuer_0", floor: 0, row: 8, col: 1, carrying: false, alive: true },
          { id: "rescuer_1", floor: 1, row: 4, col: 4, carrying: false, alive: true },
          { id: "rescuer_2", floor: 0, row: 7, col: 4, carrying: false, alive: true },
        ],
        rewards: { rescuer_0: 20, rescuer_1: 0, rescuer_2: 0 }
      }
    ]
  };
  loadReplay(fallback, "Standalone Offline Demo");
}

// Wire Event Handlers
function setupEventHandlers() {
  // Playback
  dom.playPauseBtn.addEventListener("click", togglePlayback);
  dom.prevFrameBtn.addEventListener("click", () => {
    pausePlayback();
    if (currentFrameIndex > 0) renderFrame(currentFrameIndex - 1);
  });
  dom.nextFrameBtn.addEventListener("click", () => {
    pausePlayback();
    if (currentReplay && currentFrameIndex < currentReplay.frames.length - 1) {
      renderFrame(currentFrameIndex + 1);
    }
  });

  dom.timelineSlider.addEventListener("input", (e) => {
    pausePlayback();
    renderFrame(parseInt(e.target.value, 10));
  });

  // Speed
  document.querySelectorAll(".speed-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      setSpeed(parseFloat(chip.dataset.speed));
    });
  });

  // Floor filter buttons
  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      applyFloorFilter(btn.dataset.floor);
    });
  });

  // Replay select
  dom.replaySelect.addEventListener("change", (e) => {
    if (e.target.value) {
      loadReplayFromServer(e.target.value);
    }
  });

  // Fire Mode control
  dom.fireModeControl.querySelectorAll(".segmented-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      dom.fireModeControl.querySelectorAll(".segmented-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFireMode = btn.dataset.mode;

      if (currentFireMode === "curriculum") {
        dom.fireModeCaption.textContent = "Natural stochastic fire spread from ignition points.";
      } else if (currentFireMode === "adversarial") {
        dom.fireModeCaption.textContent = "Learned PPO adversary actively blocks corridors and exits.";
      } else {
        dom.fireModeCaption.textContent = "Fire disabled (pure evacuation baseline).";
      }
    });
  });

  // Generate button
  dom.generateBtn.addEventListener("click", generateUnseenEpisode);

  // Camera reset
  dom.resetCamBtn.addEventListener("click", resetCamera);

  // Analytics Modal
  dom.openAnalyticsBtn.addEventListener("click", () => {
    dom.analyticsModal.style.display = "flex";
  });
  dom.closeModalBtn.addEventListener("click", () => {
    dom.analyticsModal.style.display = "none";
  });
  dom.analyticsModal.addEventListener("click", (e) => {
    if (e.target === dom.analyticsModal) dom.analyticsModal.style.display = "none";
  });

  // Modal Tabs
  document.querySelectorAll(".tab-btn").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");
      tab.classList.add("active");
      const targetId = `tab-${tab.dataset.tab}`;
      const targetEl = document.getElementById(targetId);
      if (targetEl) targetEl.style.display = "block";
    });
  });

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

    if (e.code === "Space") {
      e.preventDefault();
      togglePlayback();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      pausePlayback();
      if (currentFrameIndex > 0) renderFrame(currentFrameIndex - 1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      pausePlayback();
      if (currentReplay && currentFrameIndex < currentReplay.frames.length - 1) {
        renderFrame(currentFrameIndex + 1);
      }
    } else if (e.code === "KeyR") {
      resetCamera();
    } else if (e.code === "KeyG") {
      generateUnseenEpisode();
    } else if (e.code === "Escape") {
      dom.analyticsModal.style.display = "none";
    }
  });
}

// Entry point
window.addEventListener("DOMContentLoaded", () => {
  initThree();
  setupEventHandlers();
  fetchBackendData();
});

