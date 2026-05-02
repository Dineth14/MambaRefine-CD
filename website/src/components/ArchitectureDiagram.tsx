import { architectureModules, modelStats } from '../data/architecture'

const MODULE_COLORS: Record<string, { bg: string; border: string; label: string }> = {
  encoder: { bg: 'bg-blue-50', border: 'border-blue-300', label: 'text-blue-800' },
  drbi: { bg: 'bg-teal-50', border: 'border-teal-300', label: 'text-teal-800' },
  cram: { bg: 'bg-purple-50', border: 'border-purple-300', label: 'text-purple-800' },
  arf: { bg: 'bg-orange-50', border: 'border-orange-300', label: 'text-orange-800' },
  boundary: { bg: 'bg-rose-50', border: 'border-rose-300', label: 'text-rose-800' },
}

export default function ArchitectureDiagram() {
  return (
    <div className="space-y-8">
      {/* Flow diagram */}
      <div className="card p-6">
        <h3 className="text-sm font-semibold text-muted-text mb-5 uppercase tracking-wide">Pipeline Overview</h3>
        <div className="flex flex-col md:flex-row items-center gap-2 overflow-x-auto">
          {/* Inputs */}
          <div className="flex flex-col gap-2 items-center">
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-center text-xs font-medium text-slate-700 w-28">
              Image T1
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-center text-xs font-medium text-slate-700 w-28">
              Image T2
            </div>
          </div>

          <Arrow />

          {/* Encoder */}
          <Block color={MODULE_COLORS.encoder} title="Shared Encoder" subtitle="MambaVision-S" />

          <Arrow />

          {/* D-RBI */}
          <Block color={MODULE_COLORS.drbi} title="D-RBI" subtitle="Differential Region-Boundary Interaction" />

          <Arrow />

          {/* ARF-FPN */}
          <Block color={MODULE_COLORS.arf} title="ARF-FPN" subtitle="Adaptive Receptive Field" />

          <Arrow />

          {/* Boundary head */}
          <div className="flex flex-col gap-2 items-center">
            <Block color={MODULE_COLORS.boundary} title="Boundary Residual" subtitle="Correction Head" />
            <Block color={MODULE_COLORS.cram} title="CRAMLite" subtitle="Region Attention" />
          </div>

          <Arrow />

          {/* Output */}
          <div className="rounded-lg border border-slate-300 bg-white px-4 py-3 text-center text-xs font-semibold text-slate-800 w-28 shadow-sm">
            Change Map
          </div>
        </div>
      </div>

      {/* Module cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {architectureModules.map((m) => {
          const c = MODULE_COLORS[m.id] ?? MODULE_COLORS.encoder
          return (
            <div key={m.id} className={`card card-hover ${c.bg} ${c.border} border`}>
              <h4 className={`font-bold text-sm ${c.label} mb-1`}>{m.name}</h4>
              <p className="text-xs text-muted-text mb-3">{m.description}</p>
              <p className="text-xs text-slate-600 leading-relaxed">{m.description}</p>
            </div>
          )
        })}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard label="Total Params" value={modelStats.totalParams} />
        <StatCard label="Backbone Params" value={modelStats.backboneParams} />
        <StatCard label="Input Resolution" value={modelStats.inputResolution} />
        <StatCard label="Output" value={modelStats.output} />
      </div>
    </div>
  )
}

function Arrow() {
  return (
    <svg width="24" height="16" viewBox="0 0 24 16" className="text-slate-300 shrink-0 rotate-0 md:rotate-0">
      <path d="M0 8 H20 M14 2 L20 8 L14 14" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function Block({ color, title, subtitle }: { color: { bg: string; border: string; label: string }; title: string; subtitle: string }) {
  return (
    <div className={`rounded-xl border ${color.border} ${color.bg} px-4 py-3 text-center w-40 shrink-0`}>
      <p className={`text-xs font-bold ${color.label}`}>{title}</p>
      <p className="text-[10px] text-muted-text mt-0.5">{subtitle}</p>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card text-center py-4">
      <p className="text-lg font-bold text-primary">{value}</p>
      <p className="text-[11px] text-muted-text mt-0.5 font-medium">{label}</p>
    </div>
  )
}
