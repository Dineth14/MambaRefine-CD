export default function LossDiagram() {
  const terms = [
    { name: 'ℒ_bce', color: 'bg-blue-100 border-blue-300 text-blue-800', desc: 'Binary cross-entropy on final logits' },
    { name: 'ℒ_dice', color: 'bg-teal-100 border-teal-300 text-teal-800', desc: 'Dice loss on final prediction' },
    { name: 'ℒ_focal', color: 'bg-purple-100 border-purple-300 text-purple-800', desc: 'Focal loss for class imbalance' },
    { name: 'ℒ_boundary', color: 'bg-rose-100 border-rose-300 text-rose-800', desc: 'Sobel/Laplacian boundary fidelity' },
    { name: 'ℒ_coarse', color: 'bg-orange-100 border-orange-300 text-orange-800', desc: 'Auxiliary loss on coarse decoder output' },
  ]

  return (
    <div className="card p-6">
      <h3 className="text-sm font-semibold text-muted-text mb-4 uppercase tracking-wide">Training Objective</h3>

      {/* Combined loss formula */}
      <div className="bg-slate-50 rounded-lg p-4 mb-5 text-center font-mono text-sm text-slate-800 overflow-x-auto">
        ℒ = ℒ_bce + ℒ_dice + λ_f · ℒ_focal + λ_b · ℒ_boundary + λ_c · ℒ_coarse
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {terms.map((t) => (
          <div key={t.name} className={`rounded-lg border ${t.color} px-3 py-3`}>
            <p className="font-mono text-sm font-bold mb-1">{t.name}</p>
            <p className="text-xs">{t.desc}</p>
          </div>
        ))}
      </div>

      <p className="text-xs text-muted-text mt-4">
        Weights λ_f, λ_b, λ_c are set to 0.5, 0.3, 0.4 by default. The boundary auxiliary loss is applied to
        intermediate decoder outputs and suppressed at inference time.
      </p>
    </div>
  )
}
