const DATASETS = [
  {
    name: 'DSIFN-CD',
    subtitle: 'Clean image-level split used for verified DSIFN-CD evaluation',
    image: null,
    details: [
      { key: 'Split', value: '2758 train / 394 val / 789 test images' },
      { key: 'Test tiles', value: '3156' },
      { key: 'Patch size', value: '256x256' },
      { key: 'Overlap', value: '0.25' },
      { key: 'Split integrity', value: 'PASS' },
      { key: 'Leakage check', value: 'Clean split, no train/test leakage' },
    ],
    tags: ['Binary masks', 'Clean split', 'Patch inference'],
    color: 'border-primary bg-soft-blue',
    badge: 'bg-primary/10 text-primary',
    note: 'The test threshold is selected from validation, not tuned on the test set.',
  },
  {
    name: 'WHU-CD',
    subtitle: 'WHU Building Change Detection Dataset',
    image: null,
    details: [
      { key: 'Split', value: 'Standard train/val/test split' },
      { key: 'Inference', value: 'Patch-based inference' },
      { key: 'Use', value: 'Direct comparison with recent literature' },
      { key: 'Task', value: 'Binary building change detection' },
    ],
    tags: ['WHU-CD', 'Building change', 'Standard split'],
    color: 'border-secondary bg-soft-green',
    badge: 'bg-secondary/10 text-secondary',
    note: 'The test threshold is selected from validation, not tuned on the test set.',
  },
]

export default function DatasetCard() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {DATASETS.map((d) => (
        <div key={d.name} className={`card card-hover border-l-4 ${d.color}`}>
          <div className="flex items-start justify-between gap-4 mb-3">
            <div>
              <h3 className="text-lg font-bold text-main-text">{d.name}</h3>
              <p className="text-xs text-muted-text mt-0.5">{d.subtitle}</p>
            </div>
            <div className="flex flex-wrap gap-1 justify-end">
              {d.tags.map((t) => (
                <span key={t} className={`tag text-[10px] ${d.badge}`}>{t}</span>
              ))}
            </div>
          </div>

          {/* Placeholder for qualitative image */}
          <div className="w-full h-32 bg-slate-100 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-xs text-slate-400 mb-4">
            Add qualitative result here
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs mb-4">
            {d.details.map((item) => (
              <div key={item.key}>
                <dt className="font-semibold text-muted-text">{item.key}</dt>
                <dd className="text-main-text">{item.value}</dd>
              </div>
            ))}
          </dl>

          {d.note && (
            <p className="text-xs text-slate-500 bg-slate-50 rounded-md p-2 border border-slate-200">
              {d.note}
            </p>
          )}
        </div>
      ))}
    </div>
  )
}
