import { whuSorted, dsifnComparison, type ComparisonRow } from '../data/comparisons'
import { useState } from 'react'
import { AlertTriangle } from 'lucide-react'

function rankBg(rank?: 1 | 2 | 3) {
  if (rank === 1) return 'bg-amber-50'
  if (rank === 2) return 'bg-slate-50'
  if (rank === 3) return 'bg-blue-50'
  return ''
}

function rankBadge(rank?: 1 | 2 | 3) {
  if (rank === 1) return <span className="ml-1 text-[10px] font-bold text-amber-600 bg-amber-100 rounded px-1">1st</span>
  if (rank === 2) return <span className="ml-1 text-[10px] font-bold text-slate-600 bg-slate-200 rounded px-1">2nd</span>
  if (rank === 3) return <span className="ml-1 text-[10px] font-bold text-blue-600 bg-blue-100 rounded px-1">3rd</span>
  return null
}

function WHUTable({ rows }: { rows: ComparisonRow[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border shadow-card">
      <table className="result-table">
        <thead>
          <tr>
            <th>Method</th>
            <th>Year</th>
            <th>Pre (%)</th>
            <th>Rec (%)</th>
            <th>F1 (%)</th>
            <th>IoU (%)</th>
            <th>OA (%)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.method} className={`${r.isOurs ? 'font-semibold' : ''} ${rankBg(r.rank)}`}>
              <td>
                {r.method}
                {rankBadge(r.rank)}
                {r.isOurs && <span className="ml-1 text-[10px] font-bold text-primary bg-soft-blue rounded px-1">ours</span>}
              </td>
              <td className="text-muted-text">{r.year}</td>
              <td>{r.pre?.toFixed(2) ?? '—'}</td>
              <td>{r.rec?.toFixed(2) ?? '—'}</td>
              <td className="text-primary font-semibold">{r.f1.toFixed(2)}</td>
              <td>{r.iou.toFixed(2)}</td>
              <td>{r.oa?.toFixed(2) ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DSIFNTable({ rows }: { rows: ComparisonRow[] }) {
  return (
    <div>
      <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-50 border border-amber-200 mb-4 text-sm text-amber-800">
        <AlertTriangle size={16} className="mt-0.5 shrink-0" />
        <span>
          <strong>Protocol caution:</strong> Different DSIFN-CD papers use different patch protocols
          (literature uses 14400/1360/192 patch split; our results use a clean 2758/394/789 image-level split
          with 0.25 overlap tiling at inference). This table is <em>contextual</em>, not a head-to-head ranking.
        </span>
      </div>
      <div className="overflow-x-auto rounded-xl border border-border shadow-card">
        <table className="result-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Pre (%)</th>
              <th>Rec (%)</th>
              <th>F1 (%)</th>
              <th>IoU (%)</th>
              <th>OA (%)</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.method} className={r.isOurs ? 'bg-slate-50 font-semibold' : ''}>
                <td>
                  {r.method}
                  {r.isOurs && <span className="ml-1 text-[10px] font-bold text-primary bg-soft-blue rounded px-1">ours</span>}
                </td>
                <td>{r.pre?.toFixed(2) ?? '—'}</td>
                <td>{r.rec?.toFixed(2) ?? '—'}</td>
                <td className={r.isOurs ? 'text-primary' : ''}>{r.f1.toFixed(2)}</td>
                <td>{r.iou.toFixed(2)}</td>
                <td>{r.oa?.toFixed(2) ?? '—'}</td>
                <td className="text-xs text-muted-text max-w-[200px]">{r.note ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function RankingTable() {
  const [tab, setTab] = useState<'whu' | 'dsifn'>('whu')

  return (
    <div>
      <div className="flex gap-2 mb-5">
        {(['whu', 'dsifn'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              tab === t
                ? 'bg-primary text-white'
                : 'bg-white border border-border text-muted-text hover:bg-slate-50'
            }`}
          >
            {t === 'whu' ? 'WHU-CD' : 'DSIFN-CD (contextual)'}
          </button>
        ))}
      </div>
      {tab === 'whu' ? <WHUTable rows={whuSorted} /> : <DSIFNTable rows={dsifnComparison} />}
    </div>
  )
}
