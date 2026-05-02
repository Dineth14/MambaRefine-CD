const TIMELINE = [
  {
    phase: 'Baseline',
    label: 'FPN Baseline',
    what: 'Establish SimpleCNN + FPN as A0 baseline',
    f1: '76.63%',
    color: 'bg-slate-100 border-slate-300 text-slate-700',
  },
  {
    phase: 'Encoder',
    label: 'MambaVision-S',
    what: 'Replace encoder; biggest single gain',
    f1: '93.87%',
    color: 'bg-blue-100 border-blue-300 text-blue-700',
  },
  {
    phase: 'Interaction',
    label: 'D-RBI + Signed Diff',
    what: 'Add differential region-boundary interaction with signed temporal encoding',
    f1: '94.28%',
    color: 'bg-teal-100 border-teal-300 text-teal-700',
  },
  {
    phase: 'Decoder',
    label: 'ARF-FPN',
    what: 'Multi-scale adaptive receptive field decoder',
    f1: '94.36%',
    color: 'bg-orange-100 border-orange-300 text-orange-700',
  },
  {
    phase: 'Refinement',
    label: 'Boundary Residual',
    what: 'Explicit boundary correction head',
    f1: '93.59%',
    color: 'bg-rose-100 border-rose-300 text-rose-700',
  },
  {
    phase: 'Full Model',
    label: 'CRAMLite + Aux Losses',
    what: 'CRAMLite attention, coarse and boundary auxiliary losses',
    f1: '95.67%',
    color: 'bg-primary bg-opacity-10 border-primary text-primary',
    highlight: true,
  },
]

export default function Timeline() {
  return (
    <div className="relative pl-6 border-l-2 border-border space-y-6">
      {TIMELINE.map((t, i) => (
        <div key={i} className="relative">
          {/* Dot */}
          <div className={`absolute -left-[25px] w-4 h-4 rounded-full border-2 ${t.highlight ? 'bg-primary border-primary' : 'bg-white border-slate-300'}`} />
          <div className={`card card-hover border ${t.color} ${t.highlight ? 'shadow-md' : ''}`}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wide opacity-70">{t.phase}</span>
                <h4 className={`font-bold text-sm mt-0.5 ${t.highlight ? 'text-primary' : ''}`}>{t.label}</h4>
                <p className="text-xs text-muted-text mt-1">{t.what}</p>
              </div>
              <div className="text-right shrink-0">
                <p className={`text-lg font-bold ${t.highlight ? 'text-primary' : 'text-main-text'}`}>{t.f1}</p>
                <p className="text-[10px] text-muted-text">DSIFN F1</p>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
