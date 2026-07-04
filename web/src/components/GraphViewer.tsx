import { useEffect, useRef, useState } from "react";
import ForceGraph3D from "3d-force-graph";
import * as THREE from "three";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

interface MovieNode {
  id: string;
  movieId: number;
  title: string;
  genres: string[];
  imdbUrl: string | null;
  tmdbUrl: string | null;
  posterUrl: string | null;
  pagerank: number;
  degree: number;
  weightedDegree: number;
  clusterId: number | null;
  ratingCount: number;
  x?: number;
  y?: number;
  z?: number;
  fx?: number;
  fy?: number;
  fz?: number;
  __threeObj?: THREE.Object3D;
}

interface MovieEdge {
  source: string | MovieNode;
  target: string | MovieNode;
  weight: number;
  __lineObj?: THREE.Object3D & { material: THREE.Material };
}

interface GraphPayload {
  meta: Record<string, unknown>;
  nodes: MovieNode[];
  edges: MovieEdge[];
}

// Neon-leaning but not fully saturated — reads as vivid without turning into
// pure #ff00ff-style neon.
const CLUSTER_COLORS = [
  "#7dd3fc", "#a78bfa", "#f472b6", "#34d399",
  "#fbbf24", "#fb7185", "#60a5fa", "#c084fc",
];

// Fixed radius for the node shell. Positions are pinned (fx/fy/fz) via a
// Fibonacci-sphere distribution, so the silhouette is a perfectly even sphere
// regardless of degree/edges, and no continuous force simulation is needed
// (much cheaper to render — the previous edge-driven layout kept re-ticking
// every frame and was a big part of the lag).
const SPHERE_RADIUS = 320;

function fibonacciSpherePoint(index: number, total: number, radius: number) {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const y = total > 1 ? 1 - (index / (total - 1)) * 2 : 0;
  const radiusAtY = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = goldenAngle * index;
  return {
    x: Math.cos(theta) * radiusAtY * radius,
    y: y * radius,
    z: Math.sin(theta) * radiusAtY * radius,
  };
}

function clusterColor(clusterId: number | null): string {
  return clusterId != null ? CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length] : "#9ca3af";
}

function linkEndId(end: string | MovieNode): string {
  return typeof end === "string" ? end : end.id;
}

// scene.background rather than a transparent canvas + CSS gradient behind it
// (the bloom composer flattens page-level CSS to opaque black). Flat black —
// the star field lives as real 3D points (see createStarfield) so it can
// drift/parallax instead of being baked into a static texture.
function createBackdropTexture(): THREE.CanvasTexture {
  const size = 1024;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

// Sparse white points on a large shell around the sphere. Real 3D objects
// (not a painted texture) so slow rotation reads as gentle drift/parallax
// rather than a static "dead" backdrop, and the existing bloom pass gives
// them their soft neon flicker.
function createStarfield(): THREE.Points {
  const starCount = 1400;
  const positions = new Float32Array(starCount * 3);
  // Pushed well past the camera's max orbit distance so self-rotation and
  // camera parallax both read as a slow drift instead of a fast sweep.
  const minRadius = SPHERE_RADIUS * 6;
  const maxRadius = SPHERE_RADIUS * 14;
  for (let i = 0; i < starCount; i++) {
    const r = minRadius + Math.random() * (maxRadius - minRadius);
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.cos(phi);
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 1.2,
    // Fixed screen-space size (not shrunk by distance) — at this shell
    // radius, size-attenuated points render as barely-visible specks.
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
  });
  return new THREE.Points(geometry, material);
}

// One shared unit-sphere geometry, scaled per node. A "base" (no glow) and a
// "glow" (bright, cluster-colored) material per cluster — 400 unique
// geometries+materials (plus 400 poster GPU textures, when the sphere used
// poster images) was the actual cause of the earlier hover-triggered
// stutter, so this stays capped at 8 clusters x 2 states = 16 materials.
const sharedSphereGeometry = new THREE.SphereGeometry(1, 12, 12);
const planetTextureCache = new Map<string, THREE.CanvasTexture>();
const nodeMaterialCache = new Map<string, THREE.MeshPhongMaterial>();
const linkMaterialCache = new Map<string, THREE.LineBasicMaterial | THREE.MeshBasicMaterial>();

// Procedural gas-giant-style banded texture per cluster color (horizontal
// bands + wavy turbulence streaks) — this maps correctly onto a sphere's
// default equirectangular UVs (bands = latitude), unlike random swirls, and
// reads as an actual textured planet rather than a flat tinted ball.
function createPlanetTexture(hex: string): THREE.CanvasTexture {
  const width = 128;
  const height = 64;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const base = new THREE.Color(hex);

  const bandCount = 6 + Math.floor(Math.random() * 4);
  const bandHeight = height / bandCount;
  for (let i = 0; i < bandCount; i++) {
    const tint = base.clone();
    tint.lerp(new THREE.Color(Math.random() > 0.5 ? "#ffffff" : "#000000"), Math.random() * 0.3);
    ctx.fillStyle = `rgb(${tint.r * 255}, ${tint.g * 255}, ${tint.b * 255})`;
    ctx.fillRect(0, Math.round(i * bandHeight), width, Math.ceil(bandHeight) + 1);
  }

  ctx.globalAlpha = 0.25;
  for (let i = 0; i < 5; i++) {
    const y = Math.random() * height;
    ctx.strokeStyle = Math.random() > 0.5 ? "#ffffff" : "#000000";
    ctx.lineWidth = 1 + Math.random() * 2;
    ctx.beginPath();
    ctx.moveTo(0, y);
    for (let x = 0; x <= width; x += 8) {
      ctx.lineTo(x, y + Math.sin(x * 0.15 + i) * 3);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  return tex;
}

function getPlanetTexture(color: string): THREE.CanvasTexture {
  let tex = planetTextureCache.get(color);
  if (!tex) {
    tex = createPlanetTexture(color);
    planetTextureCache.set(color, tex);
  }
  return tex;
}

// glow=false: flat, unlit-looking base state (default). glow=true: bright
// cluster-colored emissive, only applied to the hovered/selected node.
function getNodeMaterial(clusterId: number | null, glow: boolean): THREE.MeshPhongMaterial {
  const color = clusterColor(clusterId);
  const key = `${color}|${glow}`;
  let mat = nodeMaterialCache.get(key);
  if (!mat) {
    mat = new THREE.MeshPhongMaterial({
      map: getPlanetTexture(color),
      emissive: new THREE.Color(color),
      emissiveIntensity: glow ? 1.1 : 0.04,
      shininess: 90,
      specular: new THREE.Color(0x999999),
    });
    nodeMaterialCache.set(key, mat);
  }
  return mat;
}

// Matches whichever renderer three-forcegraph picked for this link (thin
// Line vs a thicker tube Mesh) so swapping .material never mismatches type.
function getLinkMaterial(clusterId: number | null, isMesh: boolean) {
  const color = clusterColor(clusterId);
  const key = `${color}|${isMesh}`;
  let mat = linkMaterialCache.get(key);
  if (!mat) {
    mat = isMesh
      ? new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 });
    linkMaterialCache.set(key, mat);
  }
  return mat;
}

function buildNodeObject(node: MovieNode): THREE.Object3D {
  const radius = 3 + Math.sqrt(node.degree) * 0.9;
  const mesh = new THREE.Mesh(sharedSphereGeometry, getNodeMaterial(node.clusterId, false));
  mesh.scale.setScalar(radius);
  return mesh;
}

export default function GraphViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const hoverIdRef = useRef<string | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const nodeByIdRef = useRef<Map<string, MovieNode>>(new Map());
  const linksByNodeRef = useRef<Map<string, MovieEdge[]>>(new Map());
  const [hoverNode, setHoverNode] = useState<MovieNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<MovieNode | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [loading, setLoading] = useState(true);
  const [meta, setMeta] = useState<GraphPayload["meta"] | null>(null);

  useEffect(() => {
    const onMove = (e: MouseEvent) => setMousePos({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    // Directly mutates each node/link's own Three.js objects (via the
    // __threeObj/__lineObj refs three-forcegraph attaches) instead of going
    // through graph.refresh() — refresh() re-evaluates every node/link
    // accessor (~400 nodes + ~3300 links) on every call, which is what made
    // hovering laggy before. Direct mutation touches only the O(degree)
    // objects actually affected, so it's cheap no matter how fast the mouse
    // moves across the sphere.
    const applyNodeState = (id: string | null) => {
      if (!id) return;
      const node = nodeByIdRef.current.get(id);
      if (!node?.__threeObj) return;
      const active = id === hoverIdRef.current || id === selectedIdRef.current;
      (node.__threeObj as THREE.Mesh).material = getNodeMaterial(node.clusterId, active);
      for (const link of linksByNodeRef.current.get(id) ?? []) {
        const lineObj = link.__lineObj;
        if (!lineObj) continue;
        lineObj.visible = active;
        if (active) {
          lineObj.material = getLinkMaterial(node.clusterId, (lineObj as any).isMesh === true);
        }
      }
    };

    // Default control scheme is TrackballControls, which has no autoRotate
    // support at all (silently ignores the property) — force OrbitControls
    // so the auto-rotate-while-idle effect actually works.
    const graph = new ForceGraph3D(containerRef.current, { controlType: "orbit" })
      .backgroundColor("#000000")
      .nodeLabel(() => "")
      .nodeThreeObject((node: any) => buildNodeObject(node as MovieNode))
      .nodeThreeObjectExtend(false)
      // No physics: node positions are pinned to a Fibonacci sphere below, so
      // there's nothing to warm up/cool down and no per-frame simulation cost.
      .warmupTicks(0)
      .cooldownTicks(0)
      // All link objects are created up front (constant `true`) so their
      // __lineObj refs exist; they're hidden immediately after via direct
      // .visible mutation (see the setTimeout below), and toggled the same
      // way from then on — never through this accessor again.
      .linkVisibility(() => true)
      .linkWidth((link: any) => Math.max(0.4, (link.weight ?? 0.5) * 1.3))
      .linkColor(() => "rgba(255,255,255,0.85)")
      .onNodeHover((node: any) => {
        const n = node as MovieNode | null;
        const prevHoverId = hoverIdRef.current;
        setHoverNode(n);
        hoverIdRef.current = n ? n.id : null;
        if (prevHoverId && prevHoverId !== hoverIdRef.current) applyNodeState(prevHoverId);
        if (hoverIdRef.current) applyNodeState(hoverIdRef.current);
        // Pause auto-rotate while inspecting a node; resume once nothing is
        // hovered or selected.
        controls.autoRotate = !hoverIdRef.current && !selectedIdRef.current;
        if (containerRef.current) containerRef.current.style.cursor = n ? "pointer" : "default";
      })
      .onNodeClick((node: any) => {
        const n = node as MovieNode;
        const prevSelectedId = selectedIdRef.current;
        setSelectedNode(n);
        selectedIdRef.current = n.id;
        if (prevSelectedId && prevSelectedId !== n.id) applyNodeState(prevSelectedId);
        applyNodeState(n.id);
        controls.autoRotate = false;
        const distance = 120;
        const distRatio = 1 + distance / Math.hypot(n.x ?? 1, n.y ?? 1, n.z ?? 1);
        graph.cameraPosition(
          { x: (n.x ?? 0) * distRatio, y: (n.y ?? 0) * distRatio, z: (n.z ?? 0) * distRatio },
          { x: n.x ?? 0, y: n.y ?? 0, z: n.z ?? 0 },
          1000,
        );
      })
      .onBackgroundClick(() => {
        const prevSelectedId = selectedIdRef.current;
        setSelectedNode(null);
        selectedIdRef.current = null;
        if (prevSelectedId) applyNodeState(prevSelectedId);
        controls.autoRotate = !hoverIdRef.current;
      });

    const controls = graph.controls() as any;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    const renderer = graph.renderer();
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    graph.scene().background = createBackdropTexture();

    // Default far plane (2000) is closer than the far side of the star
    // shell (up to ~camera distance + 14 * SPHERE_RADIUS); extend it so
    // stars behind the sphere aren't clipped.
    const camera = graph.camera() as THREE.PerspectiveCamera;
    camera.far = 20000;
    camera.updateProjectionMatrix();

    const starfield = createStarfield();
    graph.scene().add(starfield);
    let starRafId = requestAnimationFrame(function animateStars() {
      starfield.rotation.y += 0.00002;
      starfield.rotation.x += 0.000006;
      starRafId = requestAnimationFrame(animateStars);
    });

    // Quarter-resolution bloom render targets: much cheaper per frame, and
    // still visually smooth at this blur radius.
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth / 3, window.innerHeight / 3),
      0.32,
      0.3,
      0.82,
    );
    graph.postProcessingComposer().addPass(bloomPass);

    fetch("/movie_graph.json")
      .then((res) => res.json())
      .then((data: GraphPayload) => {
        if (disposed) return;
        setMeta(data.meta);
        const total = data.nodes.length;
        const nodes = data.nodes.map((node, i) => {
          const { x, y, z } = fibonacciSpherePoint(i, total, SPHERE_RADIUS);
          return { ...node, x, y, z, fx: x, fy: y, fz: z };
        });
        const links = data.edges.map((e) => ({ ...e }));
        graph.graphData({ nodes, links: links as any });

        // Start further back than the library's auto-fit distance so the
        // whole sphere reads clearly the moment it appears. 3d-force-graph
        // re-aims the camera to its own auto-fit distance on the first
        // simulation tick after graphData() — setting this on the same
        // synchronous tick loses that race, so it's set again a beat later.
        const setInitialCamera = () => graph.cameraPosition({ x: 0, y: 0, z: SPHERE_RADIUS * 5.2 }, { x: 0, y: 0, z: 0 }, 0);
        setInitialCamera();

        // __threeObj / __lineObj only exist once three-forcegraph has run its
        // first digest pass, which happens on the next tick, not synchronously
        // within graphData(). Wait a beat, then hide all links and index
        // nodes/links for the direct-mutation hover/click handlers above.
        setTimeout(() => {
          if (disposed) return;
          setInitialCamera();
          const liveData = graph.graphData() as unknown as { nodes: MovieNode[]; links: MovieEdge[] };
          nodeByIdRef.current = new Map(liveData.nodes.map((n) => [n.id, n]));
          const byNode = new Map<string, MovieEdge[]>();
          for (const link of liveData.links) {
            if (link.__lineObj) link.__lineObj.visible = false;
            for (const endId of [linkEndId(link.source), linkEndId(link.target)]) {
              if (!byNode.has(endId)) byNode.set(endId, []);
              byNode.get(endId)!.push(link);
            }
          }
          linksByNodeRef.current = byNode;
          setLoading(false);
        }, 50);
      })
      .catch((err) => console.error("Failed to load movie_graph.json:", err));

    const onResize = () => graph.width(window.innerWidth).height(window.innerHeight);
    window.addEventListener("resize", onResize);
    onResize();

    return () => {
      disposed = true;
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(starRafId);
      starfield.geometry.dispose();
      (starfield.material as THREE.Material).dispose();
      graph._destructor();
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, []);

  const panelNode = selectedNode ?? hoverNode;
  const panelStyle = panelNode
    ? {
        left: Math.min(mousePos.x + 20, window.innerWidth - 300),
        top: Math.min(mousePos.y + 20, window.innerHeight - 280),
      }
    : undefined;

  return (
    <div className="relative w-full h-full">
      <div
        ref={containerRef}
        className={`absolute inset-0 transition-opacity duration-[1200ms] ease-out ${loading ? "opacity-0" : "opacity-100"}`}
      />

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="h-10 w-10 rounded-full border-2 border-white/15 border-t-white/70 animate-spin" />
        </div>
      )}

      <div className="pointer-events-none absolute top-8 left-1/2 -translate-x-1/2 text-center">
        <div className="text-2xl md:text-3xl pt-2 font-semibold tracking-wide text-white/90">
          MovieLens Discovery System
        </div>
      </div>

      <div className="pointer-events-none absolute left-16 top-1/2 -translate-y-1/2 flex flex-col gap-3">
        {meta && (
          <div className="w-60 rounded-lg border border-white/10 bg-black/50 backdrop-blur-sm p-3 text-white">
            <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">Overview</div>
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-md bg-white/5 p-2">
                <div className="text-lg font-semibold">{String(meta.n_nodes)}</div>
                <div className="text-[10px] uppercase tracking-wide text-white/50">Movies</div>
              </div>
              <div className="rounded-md bg-white/5 p-2">
                <div className="text-lg font-semibold">{String(meta.n_edges)}</div>
                <div className="text-[10px] uppercase tracking-wide text-white/50">Connections</div>
              </div>
            </div>
            <div className="mt-2 text-[11px] text-white/50">
              Edge = cosine similarity of SVD factors (sampled subgraph)
            </div>
          </div>
        )}

        <div className="w-60 rounded-lg border border-white/10 bg-black/50 backdrop-blur-sm p-3 text-white">
          <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">Clusters</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            {CLUSTER_COLORS.map((color, i) => (
              <div key={color} className="flex items-center gap-1.5 text-[11px] text-white/70">
                <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                Group {i}
              </div>
            ))}
          </div>
        </div>
      </div>

      {panelNode && (
        <div
          className="pointer-events-none fixed z-20 w-72 rounded-xl border border-white/10 bg-black/70 backdrop-blur-md p-4 text-white shadow-xl"
          style={panelStyle}
        >
          <div className="flex gap-3">
            {panelNode.posterUrl ? (
              <img src={panelNode.posterUrl} alt={panelNode.title} className="w-20 rounded-md object-cover" />
            ) : (
              <div
                className="w-20 h-28 rounded-md flex items-center justify-center text-[10px] text-white/70 text-center px-1"
                style={{ backgroundColor: clusterColor(panelNode.clusterId) }}
              >
                no poster
              </div>
            )}
            <div className="flex-1 min-w-0">
              <div className="font-semibold text-sm leading-snug">{panelNode.title}</div>
              <div className="text-xs text-white/60 mt-1">{panelNode.genres.join(" · ")}</div>
            </div>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-y-1 text-xs text-white/70">
            <span>PageRank</span>
            <span className="text-right">{panelNode.pagerank.toFixed(5)}</span>
            <span>Degree</span>
            <span className="text-right">{panelNode.degree}</span>
            <span>Ratings</span>
            <span className="text-right">{panelNode.ratingCount.toLocaleString()}</span>
            {panelNode.clusterId != null && (
              <>
                <span>Cluster</span>
                <span className="text-right">Group {panelNode.clusterId}</span>
              </>
            )}
          </div>
          <div className="pointer-events-auto mt-3 flex gap-3 text-xs">
            {panelNode.imdbUrl && (
              <a href={panelNode.imdbUrl} target="_blank" rel="noreferrer" className="text-amber-300 hover:underline">
                IMDb
              </a>
            )}
            {panelNode.tmdbUrl && (
              <a href={panelNode.tmdbUrl} target="_blank" rel="noreferrer" className="text-sky-300 hover:underline">
                TMDb
              </a>
            )}
          </div>
        </div>
      )}

      <footer className="pointer-events-none absolute bottom-0 left-0 right-0 z-10 opacity-80 my-2 min-[375px]:pl-4 md:pl-0 mt-8 w-[90%] mx-auto container lg:max-w-4xl md:max-w-2xl mb-10 flex justify-center rounded-full bg-black/20 backdrop-blur-md">
        <div className="w-full max-w-screen-xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between py-3">
          <div className="justify-center hidden md:flex md:justify-start mb-4 ml-6 md:mb-0">
            <span className="text-sm text-white/90">
              © {new Date().getFullYear()} Group 06
            </span>
          </div>
          <ul className="flex flex-wrap justify-center md:justify-end mr-6 items-center text-sm font-medium text-white/90 space-x-4 md:space-x-6 gap-20 md:gap-0">
            <li>
              <a
                href="https://github.com/BD-Grupo-06/movielens-discovery"
                target="_blank"
                rel="noreferrer"
                className="pointer-events-auto flex items-center hover:underline"
              >
                <svg
                  className="size-4 font-bold"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M9 19c-4.3 1.4 -4.3 -2.5 -6 -3m12 5v-3.5c0 -1 .1 -1.4 -.5 -2c2.8 -.3 5.5 -1.4 5.5 -6a4.6 4.6 0 0 0 -1.3 -3.2a4.2 4.2 0 0 0 -.1 -3.2s-1.1 -.3 -3.5 1.3a12.3 12.3 0 0 0 -6.2 0c-2.4 -1.6 -3.5 -1.3 -3.5 -1.3a4.2 4.2 0 0 0 -.1 3.2a4.6 4.6 0 0 0 -1.3 3.2c0 4.6 2.7 5.7 5.5 6c-.6 .6 -.6 1.2 -.5 2v3.5" />
                </svg>
                <span className="ml-1">Repository</span>
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </div>
  );
}
