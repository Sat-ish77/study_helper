"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export function WebGLBackground() {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Read accent color from CSS variable
    const style = getComputedStyle(document.documentElement);
    const accentRgb = style.getPropertyValue("--accent-rgb").trim().split(",").map(Number);
    const accentColor = new THREE.Color(accentRgb[0] / 255, accentRgb[1] / 255, accentRgb[2] / 255);

    // Create 3 low-poly floating shapes
    const shapes: { mesh: THREE.Mesh; speed: THREE.Vector3; rotSpeed: THREE.Vector3 }[] = [];
    const geometries = [
      new THREE.IcosahedronGeometry(0.6, 0),
      new THREE.OctahedronGeometry(0.5, 0),
      new THREE.TetrahedronGeometry(0.5, 0),
    ];
    const positions = [
      new THREE.Vector3(-2.5, 1, -2),
      new THREE.Vector3(2.5, -0.5, -3),
      new THREE.Vector3(0, -1.5, -1.5),
    ];

    geometries.forEach((geo, i) => {
      const mat = new THREE.MeshBasicMaterial({
        color: accentColor,
        wireframe: true,
        transparent: true,
        opacity: 0.08,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(positions[i]);
      scene.add(mesh);
      shapes.push({
        mesh,
        speed: new THREE.Vector3(
          (Math.random() - 0.5) * 0.002,
          (Math.random() - 0.5) * 0.002,
          0
        ),
        rotSpeed: new THREE.Vector3(
          Math.random() * 0.003,
          Math.random() * 0.003,
          Math.random() * 0.001
        ),
      });
    });

    // Mouse interaction
    const mouse = { x: 0, y: 0 };
    const onMouseMove = (e: MouseEvent) => {
      mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.y = -(e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener("mousemove", onMouseMove);

    // Resize
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", onResize);

    // Animation loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      shapes.forEach((s) => {
        s.mesh.rotation.x += s.rotSpeed.x;
        s.mesh.rotation.y += s.rotSpeed.y;
        s.mesh.rotation.z += s.rotSpeed.z;
        s.mesh.position.x += s.speed.x;
        s.mesh.position.y += s.speed.y;

        // Subtle mouse influence
        s.mesh.position.x += (mouse.x * 0.3 - s.mesh.position.x) * 0.001;
        s.mesh.position.y += (mouse.y * 0.3 - s.mesh.position.y) * 0.001;

        // Bounce at bounds
        if (Math.abs(s.mesh.position.x) > 4) s.speed.x *= -1;
        if (Math.abs(s.mesh.position.y) > 3) s.speed.y *= -1;
      });

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      geometries.forEach((g) => g.dispose());
      if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 pointer-events-none z-0"
      aria-hidden="true"
    />
  );
}
