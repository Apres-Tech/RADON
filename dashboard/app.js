/**
 * RADON — Autonomous Multi-Agent Rescue Simulation 3D Dashboard (Day 6 Showcase)
 * Features:
 *   - Low-height architectural walls (never hides doorways or hostages)
 *   - Continuous smooth sub-frame interpolation (polygon-game feel)
 *   - Multi-tile glowing exit zones & animated fire/hostage beacons
 *   - Hands-free continuous live game loop
 *   - Curated showcase mode (filters for high-performing real runs on unseen seeds)
 */

const CONFIG = {
  floorHeight: 4.2,
  cellSize: 1.0,
  wallHeight: 0.45, // Low walls so doorways & hostages are never obscured
  agentColors: [
    0x06b6d4, // Rescuer 0: Cyan
    0x3b82f6, // Rescuer 1: Blue
    0xa855f7, // Rescuer 2: Purple
    0xf59e0b, // Rescuer 3: Amber
    0xf43f5e, // Rescuer 4: Coral Red
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
let currentProgress = 0.0; // Continuous sub-frame float for smooth interpolation
let isPlaying = false;
let playbackSpeed = 1.0;
let baseFps = 8.0; // Base simulation steps per second
let lastTickTime = performance.now();
let currentFloorFilter = "both"; // "both", "0", "1"
let currentFireMode = "curriculum";
let currentPreset = "showcase";
let isContinuousLive = false;
let curationMode = "unfiltered"; // "unfiltered" (default: raw honest view) or "showcase" (curated best runs)

// Three.js groups
let buildingGroup = null;
let entityGroup = null;
let floorPlanes = [];
let floorMeshes = { 0: [], 1: [] };

// Entity meshes pool
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
  presetSelect: document.getElementById("preset-select"),
  continuousToggle: document.getElementById("continuous-toggle"),
  curationModeControl: document.getElementById("curation-mode-control"),
  curationCaption: document.getElementById("curation-caption"),
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
  verdictBanner: document.getElementById("verdict-banner"),
  verdictIcon: document.getElementById("verdict-icon"),
  verdictTitle: document.getElementById("verdict-title"),
  verdictSubtitle: document.getElementById("verdict-subtitle"),
  verdictStats: document.getElementById("verdict-stats"),
};

function showVerdict(type, title, subtitle, stats) {
  if (!dom.verdictBanner) return;
  dom.verdictBanner.className = `verdict-banner ${type}`;
  dom.verdictIcon.textContent = (type === 'win') ? '🏆' : (title.includes('WIPED') ? '💀' : '⏳');
  dom.verdictTitle.textContent = title;
  dom.verdictSubtitle.textContent = subtitle;
  dom.verdictStats.innerHTML = stats;
  dom.verdictBanner.style.display = 'flex';
}

function hideVerdict() {
  if (dom.verdictBanner) dom.verdictBanner.style.display = 'none';
}

// Smooth easing (Hermite smoothstep)
function smoothStep(t) {
  return t * t * (3 - 2 * t);
}

// Linear interpolation
function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Initialize 3D Engine
function initThree() {
  const width = dom.canvasContainer.clientWidth || window.innerWidth;
  const height = dom.canvasContainer.clientHeight || window.innerHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  scene.fog = new THREE.FogExp2(0x0a0e17, 0.012);

  // Camera (framed for large open levels)
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(22, 28, 28);

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
  controls.target.set(9.5, 2.0, 9.5);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xfff8ee, 0.9);
  mainLight.position.set(25, 35, 20);
  mainLight.castShadow = true;
  mainLight.shadow.mapSize.width = 2048;
  mainLight.shadow.mapSize.height = 2048;
  mainLight.shadow.camera.near = 0.5;
  mainLight.shadow.camera.far = 120;
  mainLight.shadow.camera.left = -30;
  mainLight.shadow.camera.right = 30;
  mainLight.shadow.camera.top = 30;
  mainLight.shadow.camera.bottom = -30;
  scene.add(mainLight);

  const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
  fillLight.position.set(-20, 20, -20);
  scene.add(fillLight);

  // Groups
  buildingGroup = new THREE.Group();
  entityGroup = new THREE.Group();
  scene.add(buildingGroup);
  scene.add(entityGroup);

  window.addEventListener("resize", onWindowResize);
  lastTickTime = performance.now();
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min((now - lastTickTime) / 1000.0, 0.1);
  lastTickTime = now;

  controls.update();

  // Continuous Smooth Playback Update
  if (isPlaying && currentReplay && currentReplay.frames && currentReplay.frames.length > 1) {
    const totalFrames = currentReplay.frames.length;
    currentProgress += dt * (baseFps * playbackSpeed);

    if (currentProgress >= totalFrames - 1) {
      currentProgress = totalFrames - 1;
      updateInterpolatedEntities(currentProgress);

      if (isContinuousLive) {
        // Auto-advance in live mode after short pause showing verdict banner
        pausePlayback();
        setTimeout(() => {
          if (isContinuousLive) generateUnseenEpisode(false);
        }, 2500);
      } else {
        // Loop or pause
        currentProgress = 0.0;
      }
    } else {
      updateInterpolatedEntities(currentProgress);
    }
  }

  // Animated pulse effects on Fire and Hostages
  const animTime = now * 0.005;
  fireMeshes.forEach((mesh, idx) => {
    if (mesh.visible) {
      const pulse = 1.0 + 0.18 * Math.sin(animTime * 1.5 + idx * 0.8);
      mesh.scale.set(pulse, pulse, pulse);
    }
  });

  hostageMeshes.forEach((mesh, idx) => {
    if (mesh.visible) {
      // Gentle floating bob
      mesh.position.y += Math.sin(animTime * 2.0 + idx) * 0.002;
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

// Reset camera to frame level
function resetCamera() {
  if (!currentReplay) {
    camera.position.set(22, 28, 28);
    controls.target.set(9.5, 2.0, 9.5);
  } else {
    const w = currentReplay.config.grid_w || 20;
    const h = currentReplay.config.grid_h || 20;
    const cx = (w - 1) / 2.0;
    const cz = (h - 1) / 2.0;
    const dist = Math.max(w, h) * 1.45;
    camera.position.set(cx + dist * 0.7, dist * 0.95, cz + dist * 0.85);
    controls.target.set(cx, 1.8, cz);
  }
  controls.update();
}

function showToast(message, type = "info") {
  dom.toastMessage.textContent = message;
  dom.toast.className = `toast-notification ${type}`;
  dom.toast.style.display = "flex";
  setTimeout(() => {
    dom.toast.style.display = "none";
  }, 4500);
}

// Build 3D Building Geometry with LOW walls (doorways/interior always visible!)
function buildBuilding(replayData) {
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
  const gridH = config.grid_h || 20;
  const gridW = config.grid_w || 20;

  controls.target.set((gridW - 1) / 2.0, (floors * CONFIG.floorHeight) / 4.0, (gridH - 1) / 2.0);

  // Materials
  const floorMaterial0 = new THREE.MeshStandardMaterial({
    color: 0x111827,
    roughness: 0.85,
    metalness: 0.15,
  });

  const floorMaterial1 = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.7,
    metalness: 0.25,
    transparent: true,
    opacity: 0.72,
  });

  // Low wall material: sleek concrete finish with beveled edges
  const wallMaterial = new THREE.MeshStandardMaterial({
    color: 0x334155,
    roughness: 0.6,
    metalness: 0.3,
  });

  const wallEdgeMat = new THREE.LineBasicMaterial({ color: 0x64748b, linewidth: 1 });

  // Glowing Exit Zone Material (Emerald)
  const exitMaterial = new THREE.MeshStandardMaterial({
    color: 0x10b981,
    emissive: 0x059669,
    emissiveIntensity: 0.9,
    roughness: 0.2,
  });

  // Glowing Stairs Material (Amber Gold)
  const stairsMaterial = new THREE.MeshStandardMaterial({
    color: CONFIG.stairsColor,
    emissive: 0xd97706,
    emissiveIntensity: 0.75,
    roughness: 0.3,
  });

  // 1. Build Floor Slabs
  for (let f = 0; f < floors; f++) {
    const yPos = f * CONFIG.floorHeight;

    const planeGeo = new THREE.BoxGeometry(gridW * CONFIG.cellSize, 0.12, gridH * CONFIG.cellSize);
    const planeMat = f === 0 ? floorMaterial0 : floorMaterial1;
    const planeMesh = new THREE.Mesh(planeGeo, planeMat);
    planeMesh.position.set((gridW - 1) / 2.0, yPos - 0.06, (gridH - 1) / 2.0);
    planeMesh.receiveShadow = true;
    buildingGroup.add(planeMesh);
    floorPlanes.push(planeMesh);
    floorMeshes[f].push(planeMesh);

    // Floor grid
    const gridHelper = new THREE.GridHelper(gridW, gridW, 0x475569, 0x1e293b);
    gridHelper.position.set((gridW - 1) / 2.0, yPos + 0.01, (gridH - 1) / 2.0);
    buildingGroup.add(gridHelper);
    floorMeshes[f].push(gridHelper);
  }

  // 2. Build LOW Walls (0.45 height — interior & doors always visible!)
  const mapFloors = mapData.floors || [];
  for (let f = 0; f < mapFloors.length; f++) {
    const grid = mapFloors[f];
    const yBase = f * CONFIG.floorHeight;

    for (let r = 0; r < gridH; r++) {
      for (let c = 0; c < gridW; c++) {
        if (grid[r][c] === 1) { // WALL
          const wallGeo = new THREE.BoxGeometry(0.94, CONFIG.wallHeight, 0.94);
          const wallMesh = new THREE.Mesh(wallGeo, wallMaterial);
          wallMesh.position.set(c, yBase + CONFIG.wallHeight / 2.0, r);
          wallMesh.castShadow = true;
          wallMesh.receiveShadow = true;

          const edges = new THREE.EdgesGeometry(wallGeo);
          const line = new THREE.LineSegments(edges, wallEdgeMat);
          wallMesh.add(line);

          buildingGroup.add(wallMesh);
          floorMeshes[f].push(wallMesh);
        }
      }
    }
  }

  // 3. Multi-Tile Exit ZONES on Floor 0
  const exits = mapData.exits || [];
  exits.forEach(([ef, er, ec]) => {
    const exitPadGeo = new THREE.BoxGeometry(0.94, 0.08, 0.94);
    const exitMesh = new THREE.Mesh(exitPadGeo, exitMaterial);
    exitMesh.position.set(ec, 0.04, er);
    buildingGroup.add(exitMesh);
    floorMeshes[0].push(exitMesh);

    // Exit beacon cylinder
    const beaconGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.8, 12);
    const beaconMat = new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.5 });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.set(ec, 0.44, er);
    buildingGroup.add(beacon);
    floorMeshes[0].push(beacon);
  });

  // 4. Wide Stairs Columns
  const stairs = mapData.stairs || [];
  stairs.forEach(([sr, sc]) => {
    const stairPadGeo = new THREE.BoxGeometry(0.92, 0.09, 0.92);

    const stair0 = new THREE.Mesh(stairPadGeo, stairsMaterial);
    stair0.position.set(sc, 0.045, sr);
    buildingGroup.add(stair0);
    floorMeshes[0].push(stair0);

    if (floors > 1) {
      const stair1 = new THREE.Mesh(stairPadGeo, stairsMaterial);
      stair1.position.set(sc, CONFIG.floorHeight + 0.045, sr);
      buildingGroup.add(stair1);
      floorMeshes[1].push(stair1);

      // Gold stair shaft column
      const shaftGeo = new THREE.CylinderGeometry(0.22, 0.22, CONFIG.floorHeight, 10);
      const shaftMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0xb45309, emissiveIntensity: 0.5 });
      const shaft = new THREE.Mesh(shaftGeo, shaftMat);
      shaft.position.set(sc, CONFIG.floorHeight / 2.0, sr);
      buildingGroup.add(shaft);
      floorMeshes[0].push(shaft);
      floorMeshes[1].push(shaft);
    }
  });

  applyFloorFilter(currentFloorFilter);
}

// Floor filter handler
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
}

// Initialize Dynamic Entities Pool
function initEntityPool(config) {
  while (entityGroup.children.length > 0) {
    const obj = entityGroup.children[0];
    entityGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  rescuerMeshes = {};
  hostageMeshes = [];
  fireMeshes = [];

  const nRescuers = config.n_rescuers || 5;
  const nHostages = config.n_hostages || 4;
  const maxFires = 96;

  // 1. Rescuers
  for (let i = 0; i < nRescuers; i++) {
    const rescuerId = `rescuer_${i}`;
    const agentColor = CONFIG.agentColors[i % CONFIG.agentColors.length];

    const group = new THREE.Group();

    // Body
    const bodyGeo = new THREE.CylinderGeometry(0.24, 0.28, 0.65, 12);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: agentColor,
      roughness: 0.35,
      metalness: 0.3,
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.38;
    body.castShadow = true;
    group.add(body);

    // Helmet
    const headGeo = new THREE.SphereGeometry(0.18, 12, 12);
    const headMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 0.78;
    group.add(head);

    // Visor
    const visorGeo = new THREE.BoxGeometry(0.2, 0.08, 0.12);
    const visorMat = new THREE.MeshBasicMaterial({ color: agentColor });
    const visor = new THREE.Mesh(visorGeo, visorMat);
    visor.position.set(0, 0.78, 0.15);
    group.add(visor);

    // Carrying pack (hostage carrier)
    const packGeo = new THREE.BoxGeometry(0.32, 0.32, 0.24);
    const packMat = new THREE.MeshStandardMaterial({ color: CONFIG.hostageColor, emissive: 0x059669, emissiveIntensity: 0.7 });
    const pack = new THREE.Mesh(packGeo, packMat);
    pack.position.set(0, 0.5, -0.24);
    pack.visible = false;
    group.add(pack);
    group.userData.pack = pack;

    // Fallen halo
    const haloGeo = new THREE.RingGeometry(0.26, 0.34, 16);
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

  // 2. Hostages
  for (let i = 0; i < nHostages; i++) {
    const group = new THREE.Group();

    const bodyGeo = new THREE.CylinderGeometry(0.2, 0.22, 0.55, 12);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: CONFIG.hostageColor,
      roughness: 0.3,
      emissive: 0x047857,
      emissiveIntensity: 0.45,
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.32;
    body.castShadow = true;
    group.add(body);

    const headGeo = new THREE.SphereGeometry(0.15, 12, 12);
    const headMat = new THREE.MeshStandardMaterial({ color: 0xa7f3d0 });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 0.68;
    group.add(head);

    // Medical aura ring
    const ringGeo = new THREE.RingGeometry(0.32, 0.40, 16);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x34d399, side: THREE.DoubleSide, transparent: true, opacity: 0.75 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.04;
    group.add(ring);

    entityGroup.add(group);
    hostageMeshes.push(group);
  }

  // 3. Fire Pool
  const fireGeo = new THREE.DodecahedronGeometry(0.38, 0);
  for (let i = 0; i < maxFires; i++) {
    const fireMat = new THREE.MeshStandardMaterial({
      color: CONFIG.fireColor,
      emissive: CONFIG.fireGlowColor,
      emissiveIntensity: 0.95,
      roughness: 0.2,
    });
    const fire = new THREE.Mesh(fireGeo, fireMat);
    fire.visible = false;
    entityGroup.add(fire);
    fireMeshes.push(fire);
  }
}

// Continuous Sub-Frame Interpolation Update (Smooth Polygon-Game Movement!)
function updateInterpolatedEntities(progress) {
  if (!currentReplay || !currentReplay.frames || currentReplay.frames.length === 0) return;

  const frames = currentReplay.frames;
  const config = currentReplay.config;
  const totalFrames = frames.length;

  const f0Idx = Math.min(totalFrames - 1, Math.floor(progress));
  const f1Idx = Math.min(totalFrames - 1, f0Idx + 1);
  const rawAlpha = progress - f0Idx;
  const alpha = smoothStep(Math.max(0.0, Math.min(1.0, rawAlpha)));

  const frame0 = frames[f0Idx];
  const frame1 = frames[f1Idx];

  currentFrameIndex = f0Idx;

  // 1. Interpolate Rescuers
  frame0.rescuers.forEach((r0, idx) => {
    const r1 = frame1.rescuers[idx] || r0;
    const mesh = rescuerMeshes[r0.id];
    if (mesh) {
      const isVisible = (currentFloorFilter === "both" || String(r0.floor) === currentFloorFilter);
      mesh.visible = isVisible;

      if (isVisible) {
        // Interpolate lateral position
        const posX = lerp(r0.col, r1.col, alpha);
        const posZ = lerp(r0.row, r1.row, alpha);
        const posY = lerp(r0.floor, r1.floor, alpha) * CONFIG.floorHeight;

        mesh.position.set(posX, posY, posZ);

        // Face movement direction
        const dx = r1.col - r0.col;
        const dz = r1.row - r0.row;
        if (Math.abs(dx) > 0.01 || Math.abs(dz) > 0.01) {
          mesh.rotation.y = Math.atan2(dx, dz);
        }

        // Status
        if (!r0.alive) {
          mesh.rotation.z = Math.PI / 2.2;
          mesh.position.y = posY + 0.1;
          mesh.userData.halo.visible = true;
          mesh.userData.pack.visible = false;
        } else {
          mesh.rotation.z = 0;
          mesh.userData.halo.visible = false;
          mesh.userData.pack.visible = !!r0.carrying;
        }
      }
    }
  });

  // 2. Interpolate Hostages
  frame0.hostages.forEach((h0, idx) => {
    const mesh = hostageMeshes[idx];
    if (mesh) {
      const floor = h0[0];
      const row = h0[1];
      const col = h0[2];
      const rescued = h0.length > 3 ? h0[3] : false;

      if (rescued) {
        mesh.visible = false;
      } else {
        const isVisible = (currentFloorFilter === "both" || String(floor) === currentFloorFilter);
        mesh.visible = isVisible;
        if (isVisible) {
          const yBase = floor * CONFIG.floorHeight;
          mesh.position.set(col, yBase, row);
        }
      }
    }
  });

  // 3. Fire Meshes
  fireMeshes.forEach((mesh, idx) => {
    if (idx < frame0.fire.length) {
      const fp = frame0.fire[idx];
      const floor = fp[0];
      const row = fp[1];
      const col = fp[2];
      const isVisible = (currentFloorFilter === "both" || String(floor) === currentFloorFilter);
      mesh.visible = isVisible;
      if (isVisible) {
        const yBase = floor * CONFIG.floorHeight;
        mesh.position.set(col, yBase + 0.4, row);
      }
    } else {
      mesh.visible = false;
    }
  });

  // 4. Update HUD Telemetry
  const rescuedCount = frame0.hostages.filter(h => h.length > 3 && h[3]).length;
  const survivingCount = frame0.rescuers.filter(r => r.alive).length;
  const fireCount = frame0.fire.length;

  dom.statRescued.innerHTML = `${rescuedCount} <span class="stat-unit">/ ${config.n_hostages}</span>`;
  dom.statSurvivors.innerHTML = `${survivingCount} <span class="stat-unit">/ ${config.n_rescuers}</span>`;
  dom.statFire.textContent = fireCount;
  dom.statStep.innerHTML = `${frame0.t} <span class="stat-unit">/ ${config.max_steps}</span>`;

  // Rescuer tactical cards
  let rescuerHtml = "";
  frame0.rescuers.forEach((r, i) => {
    const colorHex = `#${CONFIG.agentColors[i % CONFIG.agentColors.length].toString(16).padStart(6, "0")}`;
    let badge = "";
    if (!r.alive) badge = '<span class="rescuer-status-badge badge-dead">Dead</span>';
    else if (r.carrying) badge = '<span class="rescuer-status-badge badge-carrying">Carrying</span>';
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
  dom.timelineSlider.value = f0Idx;
  dom.currentFrameLbl.textContent = f0Idx;

  // Status tag & Unmistakable WIN / LOSE Banner Resolution
  const allRescued = (rescuedCount === config.n_hostages);
  const allDead = (survivingCount === 0);
  const isFinished = (f0Idx === totalFrames - 1) || allRescued || allDead;

  if (allRescued) {
    dom.completionTag.textContent = `✅ WIN — All ${rescuedCount}/${config.n_hostages} Evacuated!`;
    dom.completionTag.style.color = "#10b981";
    showVerdict(
      "win",
      "MISSION ACCOMPLISHED — WIN",
      `All ${rescuedCount}/${config.n_hostages} hostages safely evacuated in ${frame0.t} steps!`,
      `Rescuers Surviving: ${survivingCount}/${config.n_rescuers} &bull; Fire Deaths: ${config.n_rescuers - survivingCount}`
    );
  } else if (allDead) {
    dom.completionTag.textContent = `❌ LOSE — Squad Wiped Out (${rescuedCount}/${config.n_hostages} Saved)`;
    dom.completionTag.style.color = "#ef4444";
    showVerdict(
      "lose",
      "MISSION FAILED — LOSE (WIPED)",
      `Rescuer squad wiped out by flames. Only ${rescuedCount}/${config.n_hostages} hostages saved.`,
      `Rescuers Surviving: 0/${config.n_rescuers} &bull; Active Fire Tiles: ${fireCount}`
    );
  } else if (f0Idx === totalFrames - 1) {
    dom.completionTag.textContent = `❌ LOSE — Time Expired (${rescuedCount}/${config.n_hostages} Saved)`;
    dom.completionTag.style.color = "#ef4444";
    showVerdict(
      "lose",
      "MISSION FAILED — LOSE (TIMEOUT)",
      `Time expired at step ${frame0.t}. Saved ${rescuedCount}/${config.n_hostages} hostages.`,
      `Rescuers Surviving: ${survivingCount}/${config.n_rescuers} &bull; Active Fire Tiles: ${fireCount}`
    );
  } else {
    hideVerdict();
    dom.completionTag.textContent = `In Progress (Step ${frame0.t} / ${config.max_steps})`;
    dom.completionTag.style.color = "#38bdf8";
  }
}

// Load and Display a Replay
function loadReplay(replayData, replayTitle = "Replay") {
  currentReplay = replayData;
  currentProgress = 0.0;
  pausePlayback();
  hideVerdict();

  buildBuilding(replayData);
  initEntityPool(replayData.config);

  const frameCount = replayData.frames.length;
  dom.timelineSlider.max = Math.max(0, frameCount - 1);
  dom.timelineSlider.value = 0;
  dom.totalFramesLbl.textContent = Math.max(0, frameCount - 1);

  dom.metaCard.style.display = "block";
  dom.metaName.textContent = replayTitle;
  dom.metaSeed.textContent = replayData.config.seed !== null ? replayData.config.seed : "Procedural (Random)";

  const lastFrame = replayData.frames[frameCount - 1];
  if (lastFrame) {
    const rescued = lastFrame.hostages.filter(h => h.length > 3 && h[3]).length;
    const all = rescued === replayData.config.n_hostages;
    dom.metaOutcome.textContent = `${rescued}/${replayData.config.n_hostages} Evacuated (${all ? 'Full Clear' : 'Partial'})`;
  }

  updateInterpolatedEntities(0.0);
  resetCamera();
  startPlayback();
}

function startPlayback() {
  if (isPlaying) return;
  isPlaying = true;
  dom.playIcon.style.display = "none";
  dom.pauseIcon.style.display = "block";
  lastTickTime = performance.now();
}

function pausePlayback() {
  isPlaying = false;
  dom.playIcon.style.display = "block";
  dom.pauseIcon.style.display = "none";
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
}

// Fetch Backend Status and Replay List
async function fetchBackendData() {
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
    }
  } catch (e) {
    dom.statusDot.classList.add("offline");
    dom.backendStatusText.textContent = "Backend Offline (Viewing Mode)";
    dom.devicePill.style.display = "none";
  }

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
    loadFallbackReplay();
  }
}

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

// Live Episode Generator on Fresh Unseen Maps (Supports Curated Showcase & Unfiltered)
async function generateUnseenEpisode(isManualClick = true) {
  dom.generateBtn.disabled = true;
  dom.generateSpinner.style.display = "inline-block";
  dom.generateIcon.style.display = "none";
  dom.generateBtnText.textContent = "Simulating Unseen Map...";

  try {
    let chosenReplay = null;
    let attempts = 0;
    const maxFilterAttempts = (curationMode === "showcase") ? 3 : 1;

    while (attempts < maxFilterAttempts) {
      attempts++;
      const payload = {
        fire_mode: currentFireMode,
        seed: Math.floor(50000 + Math.random() * 49999),
        preset: currentPreset,
        max_steps: currentPreset === "showcase" ? 220 : (currentPreset === "showcase_hard" ? 400 : (currentPreset === "training_day3" ? 200 : 160)),
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
      chosenReplay = replayData;

      // In curated showcase mode, prioritize runs with >= 1 rescue or all rescued
      if (curationMode === "showcase") {
        const summary = replayData.summary || {};
        if (summary.hostages_rescued > 0 || summary.all_rescued) {
          break; // Excellent run found
        }
      } else {
        break; // Unfiltered: use first result
      }
    }

    const summary = chosenReplay.summary || {};
    const curationTag = curationMode === "showcase" ? "Showcase Filtered" : "Unfiltered Raw";
    const title = `${curationTag} (Seed ${summary.seed}, ${summary.fire_mode})`;

    loadReplay(chosenReplay, title);

    const outcomeText = summary.all_rescued
      ? `🎉 SUCCESS! All ${summary.hostages_rescued}/${summary.total_hostages} hostages evacuated in ${summary.steps} steps!`
      : `Episode completed: ${summary.hostages_rescued}/${summary.total_hostages} evacuated (${summary.rescuers_surviving}/${summary.total_rescuers} survivors).`;

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
    config: { floors: 2, grid_h: 20, grid_w: 20, n_rescuers: 5, n_hostages: 4, max_steps: 120, seed: 50001 },
    map: {
      floors: [
        [
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,6,6,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,6,6,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,1],
          [1,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,1],
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ],
        [
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,6,6,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,6,6,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ]
      ],
      exits: [[0, 17, 1], [0, 17, 2], [0, 18, 1], [0, 18, 2], [0, 17, 17], [0, 17, 18], [0, 18, 17], [0, 18, 18]],
      stairs: [[8, 8], [8, 9], [9, 8], [9, 9]]
    },
    frames: [
      {
        t: 0,
        fire: [[0, 3, 10], [1, 4, 10]],
        hostages: [[0, 6, 6, false], [0, 14, 14, false], [1, 6, 14, false], [1, 14, 6, false]],
        rescuers: [
          { id: "rescuer_0", floor: 0, row: 17, col: 2, carrying: false, alive: true },
          { id: "rescuer_1", floor: 0, row: 17, col: 3, carrying: false, alive: true },
          { id: "rescuer_2", floor: 0, row: 17, col: 16, carrying: false, alive: true },
          { id: "rescuer_3", floor: 0, row: 17, col: 17, carrying: false, alive: true },
          { id: "rescuer_4", floor: 0, row: 16, col: 9, carrying: false, alive: true },
        ],
        rewards: { rescuer_0: 0, rescuer_1: 0, rescuer_2: 0, rescuer_3: 0, rescuer_4: 0 }
      },
      {
        t: 10,
        fire: [[0, 3, 10], [0, 4, 10], [1, 4, 10], [1, 5, 10]],
        hostages: [[0, 6, 6, true], [0, 14, 14, false], [1, 6, 14, false], [1, 14, 6, false]],
        rescuers: [
          { id: "rescuer_0", floor: 0, row: 17, col: 1, carrying: false, alive: true },
          { id: "rescuer_1", floor: 1, row: 8, col: 8, carrying: false, alive: true },
          { id: "rescuer_2", floor: 0, row: 14, col: 14, carrying: true, alive: true },
          { id: "rescuer_3", floor: 0, row: 17, col: 18, carrying: false, alive: true },
          { id: "rescuer_4", floor: 1, row: 9, col: 9, carrying: false, alive: true },
        ],
        rewards: { rescuer_0: 25, rescuer_1: 0, rescuer_2: 5, rescuer_3: 0, rescuer_4: 0 }
      }
    ]
  };
  loadReplay(fallback, "Standalone Offline 20x20 Showcase");
}

function setupEventHandlers() {
  dom.playPauseBtn.addEventListener("click", togglePlayback);
  dom.prevFrameBtn.addEventListener("click", () => {
    pausePlayback();
    if (currentProgress > 0) {
      currentProgress = Math.max(0, Math.floor(currentProgress) - 1);
      updateInterpolatedEntities(currentProgress);
    }
  });
  dom.nextFrameBtn.addEventListener("click", () => {
    pausePlayback();
    if (currentReplay && currentProgress < currentReplay.frames.length - 1) {
      currentProgress = Math.min(currentReplay.frames.length - 1, Math.floor(currentProgress) + 1);
      updateInterpolatedEntities(currentProgress);
    }
  });

  dom.timelineSlider.addEventListener("input", (e) => {
    pausePlayback();
    currentProgress = parseFloat(e.target.value);
    updateInterpolatedEntities(currentProgress);
  });

  document.querySelectorAll(".speed-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      setSpeed(parseFloat(chip.dataset.speed));
    });
  });

  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      applyFloorFilter(btn.dataset.floor);
    });
  });

  dom.replaySelect.addEventListener("change", (e) => {
    if (e.target.value) {
      loadReplayFromServer(e.target.value);
    }
  });

  dom.presetSelect.addEventListener("change", (e) => {
    currentPreset = e.target.value;
  });

  dom.continuousToggle.addEventListener("change", (e) => {
    isContinuousLive = e.target.checked;
    showToast(isContinuousLive ? "Continuous Live Game enabled: auto-advances episodes." : "Continuous Live Game disabled.", "info");
  });

  dom.curationModeControl.querySelectorAll(".segmented-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      dom.curationModeControl.querySelectorAll(".segmented-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      curationMode = btn.dataset.curation;

      if (curationMode === "showcase") {
        dom.curationCaption.textContent = "Filtered for presentation: showing runs where at least 1 rescue occurred. Raw policy win rate: 6.7%.";
        dom.curationCaption.style.color = "#f59e0b";
      } else {
        dom.curationCaption.textContent = "Raw unbiased policy execution on fresh held-out maps (no filtering). Real win rate: 6.7%.";
        dom.curationCaption.style.color = "#34d399";
      }
    });
  });

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

  dom.generateBtn.addEventListener("click", () => generateUnseenEpisode(true));
  dom.resetCamBtn.addEventListener("click", resetCamera);

  dom.openAnalyticsBtn.addEventListener("click", () => {
    dom.analyticsModal.style.display = "flex";
  });
  dom.closeModalBtn.addEventListener("click", () => {
    dom.analyticsModal.style.display = "none";
  });
  dom.analyticsModal.addEventListener("click", (e) => {
    if (e.target === dom.analyticsModal) dom.analyticsModal.style.display = "none";
  });

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

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

    if (e.code === "Space") {
      e.preventDefault();
      togglePlayback();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      pausePlayback();
      if (currentProgress > 0) {
        currentProgress = Math.max(0, Math.floor(currentProgress) - 1);
        updateInterpolatedEntities(currentProgress);
      }
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      pausePlayback();
      if (currentReplay && currentProgress < currentReplay.frames.length - 1) {
        currentProgress = Math.min(currentReplay.frames.length - 1, Math.floor(currentProgress) + 1);
        updateInterpolatedEntities(currentProgress);
      }
    } else if (e.code === "KeyR") {
      resetCamera();
    } else if (e.code === "KeyG") {
      generateUnseenEpisode(true);
    } else if (e.code === "Escape") {
      dom.analyticsModal.style.display = "none";
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initThree();
  setupEventHandlers();
  fetchBackendData();
});
