'use client';

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { fetchAssetPipelineFileBlob } from '@/lib/api/sam3dAssetPipelineClient';

export type PreviewSource = 'glb' | 'mujoco_visual';

export interface AssetInteractivePreviewProps {
  jobId: string;
  objectGlbPath?: string | null;
  mujocoVisualObjPath?: string | null;
  mujocoCollisionObjPath?: string | null;
}

const PREVIEW_HEIGHT = 'min(520px, 48vh)';
const PREVIEW_MIN_HEIGHT = 420;

type NormalizationTransform = {
  center: THREE.Vector3;
  maxDim: number;
};

function fitCameraToModel(
  maxDim: number,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls
): void {
  const distance = maxDim * 2.2;
  camera.position.set(distance * 0.75, distance * 0.55, distance * 0.85);
  camera.near = Math.max(maxDim / 100, 0.001);
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
}

/** Apply a single centering transform to modelRoot based on the reference mesh local bbox. */
function normalizeModelRoot(
  modelRoot: THREE.Group,
  referenceObject: THREE.Object3D,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls
): NormalizationTransform {
  const box = new THREE.Box3().setFromObject(referenceObject);
  if (box.isEmpty()) {
    modelRoot.position.set(0, 0, 0);
    fitCameraToModel(0.001, camera, controls);
    return { center: new THREE.Vector3(), maxDim: 0.001 };
  }

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.001);

  modelRoot.position.set(-center.x, -center.y, -center.z);
  fitCameraToModel(maxDim, camera, controls);

  return { center, maxDim };
}

function applyNeutralMaterial(object: THREE.Object3D, material: THREE.Material) {
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      (child as THREE.Mesh).material = material;
    }
  });
}

function applyCollisionMaterial(object: THREE.Object3D) {
  applyNeutralMaterial(
    object,
    new THREE.MeshStandardMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.35,
      metalness: 0.05,
      roughness: 0.85,
      depthWrite: false,
    })
  );
}

async function loadModelFromBlob(
  blob: Blob,
  kind: 'glb' | 'obj'
): Promise<{ object: THREE.Object3D; objectUrl: string }> {
  const objectUrl = URL.createObjectURL(blob);
  try {
    if (kind === 'glb') {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(objectUrl);
      return { object: gltf.scene, objectUrl };
    }
    const loader = new OBJLoader();
    const object = await loader.loadAsync(objectUrl);
    return { object, objectUrl };
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw error;
  }
}

export function AssetInteractivePreview({
  jobId,
  objectGlbPath,
  mujocoVisualObjPath,
  mujocoCollisionObjPath,
}: AssetInteractivePreviewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const frameRef = useRef<number | null>(null);
  const modelRootRef = useRef<THREE.Group | null>(null);
  const objectUrlsRef = useRef<string[]>([]);

  const [previewSource, setPreviewSource] = useState<PreviewSource>('glb');
  const [showCollisionOverlay, setShowCollisionOverlay] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const availability = useMemo(
    () => ({
      glb: Boolean(objectGlbPath),
      mujocoVisual: Boolean(mujocoVisualObjPath),
      mujocoCollision: Boolean(mujocoCollisionObjPath),
    }),
    [objectGlbPath, mujocoVisualObjPath, mujocoCollisionObjPath]
  );

  const defaultSource = useMemo((): PreviewSource => {
    if (availability.glb) return 'glb';
    if (availability.mujocoVisual) return 'mujoco_visual';
    return 'glb';
  }, [availability]);

  const collisionOverlayEnabled =
    previewSource === 'mujoco_visual' && showCollisionOverlay && availability.mujocoCollision;

  useEffect(() => {
    setPreviewSource(defaultSource);
  }, [defaultSource, jobId]);

  useEffect(() => {
    if (previewSource === 'glb' && showCollisionOverlay) {
      setShowCollisionOverlay(false);
    }
  }, [previewSource, showCollisionOverlay]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#f1f5f9');
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
    fillLight.position.set(-4, 2, -3);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(2, 20, 0xcbd5e1, 0xe2e8f0);
    grid.position.y = -0.001;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(0.25));

    const modelRoot = new THREE.Group();
    scene.add(modelRoot);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    controlsRef.current = controls;
    modelRootRef.current = modelRoot;

    const resize = () => {
      const width = container.clientWidth || 640;
      const height = container.clientHeight || 420;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frameRef.current = requestAnimationFrame(animate);
    };
    animate();

    const clearModelRoot = () => {
      for (const url of objectUrlsRef.current) URL.revokeObjectURL(url);
      objectUrlsRef.current = [];
      modelRoot.clear();
      modelRoot.position.set(0, 0, 0);
    };

    const loadModels = async () => {
      setLoading(true);
      setError(null);
      clearModelRoot();

      let relPath: string | null = null;
      let kind: 'glb' | 'obj' = 'glb';

      if (previewSource === 'glb' && objectGlbPath) {
        relPath = objectGlbPath;
        kind = 'glb';
      } else if (previewSource === 'mujoco_visual' && mujocoVisualObjPath) {
        relPath = mujocoVisualObjPath;
        kind = 'obj';
      } else {
        setError('当前预览源不可用');
        setLoading(false);
        return;
      }

      try {
        const mainBlob = await fetchAssetPipelineFileBlob(jobId, relPath);
        if (cancelled) return;
        const { object: mainObject, objectUrl: mainUrl } = await loadModelFromBlob(mainBlob, kind);
        if (cancelled) {
          URL.revokeObjectURL(mainUrl);
          return;
        }
        objectUrlsRef.current.push(mainUrl);

        if (kind === 'obj') {
          applyNeutralMaterial(
            mainObject,
            new THREE.MeshStandardMaterial({ color: 0xbfc7d4, metalness: 0.15, roughness: 0.65 })
          );
        }

        modelRoot.add(mainObject);

        const transform = normalizeModelRoot(modelRoot, mainObject, camera, controls);

        let collisionObject: THREE.Object3D | null = null;
        if (collisionOverlayEnabled && mujocoCollisionObjPath) {
          const collisionBlob = await fetchAssetPipelineFileBlob(jobId, mujocoCollisionObjPath);
          if (cancelled) return;
          const { object: collisionMesh, objectUrl: collisionUrl } = await loadModelFromBlob(
            collisionBlob,
            'obj'
          );
          if (cancelled) {
            URL.revokeObjectURL(collisionUrl);
            return;
          }
          objectUrlsRef.current.push(collisionUrl);
          applyCollisionMaterial(collisionMesh);
          modelRoot.add(collisionMesh);
          collisionObject = collisionMesh;
        }

        const visualBox = new THREE.Box3().setFromObject(mainObject);
        const collisionBox = collisionObject ? new THREE.Box3().setFromObject(collisionObject) : null;

        console.debug('[AssetInteractivePreview] model transform', {
          source: previewSource,
          visualBox: {
            min: visualBox.min.toArray(),
            max: visualBox.max.toArray(),
          },
          collisionBox: collisionBox
            ? { min: collisionBox.min.toArray(), max: collisionBox.max.toArray() }
            : null,
          modelRootPosition: modelRoot.position.toArray(),
          center: transform.center.toArray(),
          maxDim: transform.maxDim,
          showCollision: collisionOverlayEnabled,
        });

      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : '3D 模型加载失败');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void loadModels();

    return () => {
      cancelled = true;
      ro.disconnect();
      if (frameRef.current != null) cancelAnimationFrame(frameRef.current);
      clearModelRoot();
      controls.dispose();
      renderer.dispose();
      container.innerHTML = '';
      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      modelRootRef.current = null;
    };
  }, [
    jobId,
    previewSource,
    collisionOverlayEnabled,
    objectGlbPath,
    mujocoVisualObjPath,
    mujocoCollisionObjPath,
  ]);

  const panelStyle: CSSProperties = {
    position: 'relative',
    width: '100%',
    height: PREVIEW_HEIGHT,
    minHeight: PREVIEW_MIN_HEIGHT,
    borderRadius: 12,
    border: '1px solid #e2e8f0',
    background: '#f8fafc',
    overflow: 'hidden',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
        {availability.glb ? (
          <SourceButton active={previewSource === 'glb'} onClick={() => setPreviewSource('glb')}>
            GLB
          </SourceButton>
        ) : null}
        {availability.mujocoVisual ? (
          <SourceButton
            active={previewSource === 'mujoco_visual'}
            onClick={() => setPreviewSource('mujoco_visual')}
          >
            MuJoCo Visual
          </SourceButton>
        ) : null}
        {previewSource === 'mujoco_visual' && availability.mujocoCollision ? (
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#475569' }}>
            <input
              type="checkbox"
              checked={showCollisionOverlay}
              onChange={(e) => setShowCollisionOverlay(e.target.checked)}
            />
            显示碰撞体
          </label>
        ) : null}
      </div>

      <div style={panelStyle}>
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
        {loading ? (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(248,250,252,0.72)',
              fontSize: 13,
              color: '#64748b',
            }}
          >
            加载 3D 模型…
          </div>
        ) : null}
        {error ? (
          <div
            style={{
              position: 'absolute',
              left: 12,
              right: 12,
              bottom: 12,
              padding: '8px 10px',
              borderRadius: 8,
              background: '#fef2f2',
              color: '#b91c1c',
              fontSize: 12,
            }}
          >
            {error}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function SourceButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: '6px 12px',
        fontSize: 12,
        fontWeight: 500,
        borderRadius: 999,
        border: active ? '1px solid #2563eb' : '1px solid #cbd5e1',
        background: active ? '#eff6ff' : '#fff',
        color: active ? '#1d4ed8' : '#475569',
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  );
}
