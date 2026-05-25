import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { ablationRows } from '../data/ablations'

const COLORS = ['#94a3b8', '#3b82f6', '#22c55e', '#f97316', '#a78bfa', '#f43f5e', '#2563eb']

export default function AblationChart() {
  const data = ablationRows.map((r, i) => ({
    name: r.id,
    f1: r.f1,
    iou: r.iou,
    label: r.label,
    isLast: i === ablationRows.length - 1,
  }))

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold text-muted-text mb-3">F1 Score by ablation step</p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} barCategoryGap="30%">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[60, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(val: number) => [`${val.toFixed(2)}%`, 'F1']}
              labelFormatter={(label: string) => data.find((d) => d.name === label)?.label ?? label}
            />
            <ReferenceLine y={95} stroke="#2563eb" strokeDasharray="4 4" label={{ value: '95%', fill: '#2563eb', fontSize: 11 }} />
            <Bar dataKey="f1" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.isLast ? '#2563eb' : COLORS[i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <p className="text-sm font-semibold text-muted-text mb-3">IoU Score by ablation step</p>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} barCategoryGap="30%">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[55, 95]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(val: number) => [`${val.toFixed(2)}%`, 'IoU']}
              labelFormatter={(label: string) => data.find((d) => d.name === label)?.label ?? label}
            />
            <Bar dataKey="iou" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.isLast ? '#0f766e' : '#94a3b8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed table */}
      <div className="overflow-x-auto rounded-xl border border-border shadow-card">
        <table className="result-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Configuration</th>
              <th>Pre (%)</th>
              <th>Rec (%)</th>
              <th>F1 (%)</th>
              <th>IoU (%)</th>
              <th>OA (%)</th>
              <th>Params (M)</th>
            </tr>
          </thead>
          <tbody>
            {ablationRows.map((r, i) => (
              <tr key={r.id} className={i === ablationRows.length - 1 ? 'bg-soft-blue font-semibold' : ''}>
                <td className="font-mono text-xs font-semibold">{r.id}</td>
                <td className="text-xs max-w-[200px]">{r.description}</td>
                <td>{r.pre.toFixed(2)}</td>
                <td>{r.rec.toFixed(2)}</td>
                <td className="text-primary font-semibold">{r.f1.toFixed(2)}</td>
                <td>{r.iou.toFixed(2)}</td>
                <td>{r.oa.toFixed(2)}</td>
                <td className="font-mono text-xs">{r.params}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
