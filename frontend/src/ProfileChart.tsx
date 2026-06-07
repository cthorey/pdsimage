// Minimal dependency-free SVG line chart for an elevation profile.

interface Props {
  z: number[];
  width?: number;
  height?: number;
}

export default function ProfileChart({ z, width = 480, height = 200 }: Props) {
  if (z.length === 0) return null;
  const pad = 36;
  const zmin = Math.min(...z);
  const zmax = Math.max(...z);
  const span = zmax - zmin || 1;

  const points = z
    .map((v, i) => {
      const x = pad + (i / (z.length - 1)) * (width - 2 * pad);
      const y = height - pad - ((v - zmin) / span) * (height - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="profile-chart">
      <rect x={pad} y={pad / 2} width={width - 2 * pad} height={height - 1.5 * pad} fill="#0b1021" stroke="#2a3b6b" />
      <polyline points={points} fill="none" stroke="#7fd1ff" strokeWidth={2} />
      <text x={4} y={pad / 2 + 6} fill="#9fb3d1" fontSize={11}>{zmax.toFixed(0)} m</text>
      <text x={4} y={height - pad + 4} fill="#9fb3d1" fontSize={11}>{zmin.toFixed(0)} m</text>
      <text x={width / 2} y={height - 6} fill="#9fb3d1" fontSize={11} textAnchor="middle">
        distance →
      </text>
    </svg>
  );
}
