import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label, ReferenceLine,
} from 'recharts'
import { backboneScaling } from '../data/results'

const COLORS: Record<string, string> = {
  'MambaVision-T (tiny)': '#94a3b8',
  'MambaVision-S (small) ★': '#2563eb',
  'MambaVision-B (base)': '#0f766e',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CustomDot = (props: any) => {
  const { cx, cy, payload } = props
  const color = COLORS[payload.label] ?? '#2563eb'
  return (
    <g>
      <circle cx={cx} cy={cy} r={10} fill={color} opacity={0.85} />
      <text x={cx} y={cy - 14} textAnchor="middle" fontSize={11} fill="#475569">
        {payload.label}
      </text>
    </g>
  )
}

export default function ComparisonChart() {
  const data = backboneScaling.map((r) => ({ ...r, x: r.params, y: r.f1 }))

  return (
    <div>
      <p className="text-sm font-semibold text-muted-text mb-3">F1 vs. Parameter count (backbone scaling)</p>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" dataKey="x" domain={[40, 120]} tick={{ fontSize: 11 }}>
            <Label value="Parameters (M)" offset={-10} position="insideBottom" fontSize={12} fill="#475569" />
          </XAxis>
          <YAxis type="number" dataKey="y" domain={[94, 96.5]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(val: number, name: string) => [`${val.toFixed(2)}${name === 'y' ? '%' : 'M'}`, name === 'y' ? 'F1' : 'Params']}
            labelFormatter={() => ''}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          content={({ payload }: any) => {
              if (!payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="bg-white border border-border rounded-lg px-3 py-2 shadow text-sm">
                  <p className="font-semibold">{d.label}</p>
                  <p className="text-muted-text">Params: {d.params}M</p>
                  <p className="text-primary">F1: {d.f1.toFixed(2)}%</p>
                  <p className="text-secondary">IoU: {d.iou.toFixed(2)}%</p>
                  {d.note && <p className="text-xs text-muted-text mt-1">{d.note}</p>}
                </div>
              )
            }}
          />
          <ReferenceLine y={95.67} stroke="#2563eb" strokeDasharray="4 4" />
          <Scatter data={data} shape={<CustomDot />} />
        </ScatterChart>
      </ResponsiveContainer>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-border shadow-card mt-6">
        <table className="result-table">
          <thead>
            <tr>
              <th>Backbone</th>
              <th>Params (M)</th>
              <th>F1 (%)</th>
              <th>IoU (%)</th>
              <th>OA (%)</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {backboneScaling.map((r) => (
              <tr key={r.label} className={r.label.includes('★') ? 'bg-soft-blue font-semibold' : ''}>
                <td>{r.label}</td>
                <td className="font-mono text-xs">{r.params}</td>
                <td className="text-primary font-semibold">{r.f1.toFixed(2)}</td>
                <td>{r.iou.toFixed(2)}</td>
                <td>{r.oa.toFixed(2)}</td>
                <td className="text-xs text-muted-text">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
