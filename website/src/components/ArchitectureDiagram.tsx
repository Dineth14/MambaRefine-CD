import { architectureModules, modelStats } from '../data/architecture'

const MODULE_COLORS: Record<string, { bg: string; border: string; label: string }> = {
  encoder: { bg: 'bg-blue-50', border: 'border-blue-300', label: 'text-blue-800' },
  drbi: { bg: 'bg-teal-50', border: 'border-teal-300', label: 'text-teal-800' },
  cram: { bg: 'bg-cyan-50', border: 'border-cyan-300', label: 'text-cyan-800' },
  arf: { bg: 'bg-orange-50', border: 'border-orange-300', label: 'text-orange-800' },
  boundary: { bg: 'bg-rose-50', border: 'border-rose-300', label: 'text-rose-800' },
}

export default function ArchitectureDiagram() {
  return (
    <div className="space-y-8">
      {/* Flow diagram */}
      <div className="card p-6">
        <h3 className="text-sm font-semibold text-muted-text mb-5 uppercase tracking-wide">Pipeline Overview</h3>
        <div className="flex flex-col xl:flex-row items-center gap-2 overflow-x-auto pb-2">
          {/* Inputs */}
          <div className="flex flex-col gap-2 items-center">
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-center text-xs font-medium text-slate-700 w-28">
              I1 pre-change
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-center text-xs font-medium text-slate-700 w-28">
              I2 post-change
            </div>
          </div>

          <Arrow />

          {/* Encoder */}
          <Block color={MODULE_COLORS.encoder} title="Shared Encoder" subtitle="E(I1), E(I2) -> F0...F3" tip="Shared weights keep feature spaces aligned across the two dates." />

          <Arrow />

          {/* D-RBI */}
          <Block color={MODULE_COLORS.drbi} title="D-RBI at each scale" subtitle="[F1,F2,abs diff,signed diff]" tip="Builds explicit temporal evidence and separates region and boundary streams." />

          <Arrow />

          <div className="flex flex-col gap-2 items-center">
            <Block color={MODULE_COLORS.cram} title="Region streams" subtitle="CRAMLite attention" tip="Lightweight residual spatial attention enhances likely change regions." />
            <Block color={MODULE_COLORS.arf} title="ARF-FPN" subtitle="d=1,2,4,8 -> P_c" tip="Parallel dilated branches adapt receptive field before coarse prediction." />
          </div>

          <Arrow />

          {/* Boundary head */}
          <div className="flex flex-col gap-2 items-center">
            <Block color={MODULE_COLORS.boundary} title="Boundary stream B0" subtitle="+ Sobel uncertainty" tip="Uses the finest boundary stream and uncertainty from the coarse map." />
            <Block color={MODULE_COLORS.boundary} title="Bounded residual" subtitle="P_f = P_c + 0.1 tanh(delta)" tip="The bounded correction prevents boundary refinement from overriding the coarse prediction." />
          </div>

          <Arrow />

          {/* Output */}
          <div className="rounded-lg border border-slate-300 bg-white px-4 py-3 text-center text-xs font-semibold text-slate-800 w-28 shadow-sm">
            sigmoid(P_f) binary map
          </div>
        </div>
        <div className="mt-5 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-3 text-xs text-muted-text">
          For each encoder scale, D-RBI receives F1^s, F2^s, |F2^s - F1^s|, and F2^s - F1^s. Region streams feed CRAMLite and ARF-FPN to produce the coarse map P_c. The finest boundary stream B0, P_c, and Sobel uncertainty produce a small residual correction before sigmoid thresholding.
        </div>
      </div>

      {/* Module cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {architectureModules.map((m) => {
          const c = MODULE_COLORS[m.id] ?? MODULE_COLORS.encoder
          return (
            <div key={m.id} className={`card card-hover ${c.bg} ${c.border} border`}>
              <h4 className={`font-bold text-sm ${c.label} mb-1`}>{m.name}</h4>
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

function Block({ color, title, subtitle, tip }: { color: { bg: string; border: string; label: string }; title: string; subtitle: string; tip?: string }) {
  return (
    <div title={tip} className={`rounded-xl border ${color.border} ${color.bg} px-4 py-3 text-center w-44 shrink-0 transition-transform hover:-translate-y-0.5`}>
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
