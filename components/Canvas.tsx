import * as THREE from 'three';
import { useMemo, useRef } from 'react';
import { MeshLineGeometry, MeshLineMaterial } from 'meshline';
import { extend, Canvas, useFrame } from '@react-three/fiber';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { easing } from 'maath';

extend({ MeshLineGeometry, MeshLineMaterial });

function HeaderCanvas() {
  const dash = 0.9;
  const count = 50;
  const radius = 50;

  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 90 }}>
      <color attach="background" args={['#18181b']} />
      <Lines
        dash={dash}
        count={count}
        radius={radius}
        colors={[[10, 0.5, 2], [1, 2, 10], '#c57fe4', '#ff0080']}
      />
      <Rig />
      <EffectComposer>
        <Bloom mipmapBlur luminanceThreshold={1} radius={0.6} />
      </EffectComposer>
    </Canvas>
  );
}

function Lines({
  dash,
  count,
  colors,
  radius = 50,
  rand = THREE.MathUtils.randFloatSpread
}) {
  const lines = useMemo(() => {
    return Array.from({ length: count }, () => {
      const pos = new THREE.Vector3(rand(radius), rand(radius), rand(radius));
      const points = Array.from({ length: 70 }, () =>
        pos
          .add(new THREE.Vector3(rand(radius), rand(radius), rand(radius)))
          .clone()
      );

      const curve = new THREE.CatmullRomCurve3(points).getPoints(300);
      return {
        color: colors[parseInt(colors.length * Math.random())],
        width: Math.max(radius / 100, (radius / 50) * Math.random()),
        speed: Math.max(0.1, 0.64 * Math.random()),
        curve: curve.flatMap(point => point.toArray())
      };
    });
  }, [colors, count, radius, rand]);
  return lines.map((props, index) => (
    <Fatline key={index} dash={dash} {...props} />
  ));
}

function Fatline({ curve, width, color, speed, dash }) {
  const ref = useRef();
  useFrame(
    (state, delta) => (ref.current.material.dashOffset -= (delta * speed) / 10)
  );
  return (
    <mesh ref={ref}>
      <meshLineGeometry points={curve} />
      <meshLineMaterial
        transparent
        lineWidth={width / 4}
        color={color}
        depthWrite={false}
        dashArray={0.25}
        dashRatio={dash}
        toneMapped={false}
      />
    </mesh>
  );
}

const rotation = {
  x: 0,
  y: 0,
  z: 0
};

function Rig({ radius = 20 }) {
  useFrame((state, dt) => {
    easing.damp3(
      state.camera.position,
      [
        Math.sin(state.pointer.x) * radius,
        Math.atan(state.pointer.y) * radius,
        Math.cos(state.pointer.x) * radius
      ],
      0.25,
      dt
    );

    state.camera.lookAt(0, 0, 0);
  });

  useFrame((state, dt) => {
    rotation.y += 0.0003;
    rotation.x += 0.0007;
    rotation.z += 0.0005;

    state.camera.rotation.set(rotation.x, rotation.y, rotation.z);
  });
}

export default HeaderCanvas;
