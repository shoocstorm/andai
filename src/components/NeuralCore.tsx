// The standby emblem: a luminous core in a wireframe sphere, ringed by two
// orbits and a slow instrument dial. Pure decoration (aria-hidden);
// it has no frame so its glow fades into the ambient backdrop.
import { useReducedMotion } from 'motion/react';
import { useId } from 'react';

const ORBITS = [
  { tilt: -28, dur: 8 },
  { tilt: 52, dur: 11 },
];
const ORBIT_RX = 80;
const ORBIT_RY = 22;
// An ellipse as a path, so a node can ride it with animateMotion.
const ORBIT_PATH = `M ${-ORBIT_RX} 0 A ${ORBIT_RX} ${ORBIT_RY} 0 1 1 ${ORBIT_RX} 0 A ${ORBIT_RX} ${ORBIT_RY} 0 1 1 ${-ORBIT_RX} 0`;
const SPHERE_R = 36;
const TICKS = Array.from({ length: 24 }, (_, i) => i);

export function NeuralCore({ size = 176 }: { size?: number }) {
  const id = useId().replace(/:/g, '');
  const still = useReducedMotion();
  return (
    <div className="nc" style={{ width: size, height: size }} aria-hidden="true">
      <i className="nc-halo" />
      <svg viewBox="-100 -100 200 200" width={size} height={size}>
        <defs>
          <radialGradient id={`${id}-heart`}>
            <stop offset="0%" className="nc-stop-hot" />
            <stop offset="35%" className="nc-stop-blue" />
            <stop offset="75%" className="nc-stop-violet" stopOpacity="0.55" />
            <stop offset="100%" className="nc-stop-violet" stopOpacity="0" />
          </radialGradient>
          <linearGradient id={`${id}-orbit`} x1="0" x2="1">
            <stop offset="0%" className="nc-stop-blue" stopOpacity="0" />
            <stop offset="50%" className="nc-stop-blue" stopOpacity="0.9" />
            <stop offset="100%" className="nc-stop-violet" stopOpacity="0.2" />
          </linearGradient>
        </defs>

        {/* instrument dial: a tick every 15°, a longer one every 90°, slowly turning */}
        <g className="nc-dial">
          {TICKS.map((i) => (
            <line key={i} x1="0" y1={-96} x2="0" y2={i % 6 ? -92 : -87} transform={`rotate(${i * 15})`} />
          ))}
        </g>
        {/* a scanning arc just inside the dial */}
        <circle className="nc-scan" r="84" pathLength="100" />

        {/* gyroscope orbits, each with one node riding it */}
        {ORBITS.map((o, i) => (
          <g key={o.tilt} transform={`rotate(${o.tilt})`}>
            <path className="nc-orbit" d={ORBIT_PATH} stroke={`url(#${id}-orbit)`} />
            <circle className="nc-node" r="2.6">
              {!still && <animateMotion dur={`${o.dur}s`} begin={`${-i * 2}s`} repeatCount="indefinite" path={ORBIT_PATH} />}
            </circle>
          </g>
        ))}

        {/* wireframe sphere: its outline and two meridians that turn */}
        <g className="nc-sphere">
          <circle r={SPHERE_R} />
          {[0, 1].map((i) => (
            <ellipse key={i} className="nc-meridian" rx={SPHERE_R} ry={SPHERE_R} style={{ animationDelay: `${-i * 3}s` }} />
          ))}
        </g>

        {/* the core */}
        <circle className="nc-heart" r="26" fill={`url(#${id}-heart)`} />
        <circle className="nc-spark" r="4.5" />
      </svg>
    </div>
  );
}
