const DATASETS = [
  {
    name: 'DSIFN-CD',
    subtitle: 'Dataset for Individual Change Detection and Semantic Segmentation',
    image: null,
    details: [
      { key: 'Source', value: 'Google Earth imagery, 5 cities' },
      { key: 'Total pairs', value: '3940 image pairs' },
      { key: 'Our split', value: '2758 train / 394 val / 789 test' },
      { key: 'Image size', value: '512 × 512 pixels' },
      { key: 'Inference tiling', value: '256 × 256, 0.25 overlap' },
      { key: 'Threshold', value: '0.60 (sweep-optimized)' },
    ],
    tags: ['Google Earth', 'Urban Change', 'Multi-Class'],
    color: 'border-primary bg-soft-blue',
    badge: 'bg-primary/10 text-primary',
    note: 'Our clean split verifies zero cross-split overlap. Literature patch-level splits (14400/1360/192) are maintained in parallel for Mamba-CD comparison.',
  },
  {
    name: 'WHU-CD',
    subtitle: 'WHU Building Change Detection Dataset',
    image: null,
    details: [
      { key: 'Source', value: 'Aerial imagery, Christchurch, NZ' },
      { key: 'Total pairs', value: '1 high-res pair tiled to 8189 patches' },
      { key: 'Standard split', value: '6096 train / 762 val / 1331 test' },
      { key: 'Image size', value: '256 × 256 pixels' },
      { key: 'Threshold', value: '0.55 (sweep-optimized)' },
    ],
    tags: ['Aerial', 'Building Change', 'Single Scene'],
    color: 'border-secondary bg-soft-green',
    badge: 'bg-secondary/10 text-secondary',
    note: 'Standard WHU-CD split; the full high-resolution aerial image is tiled. Our split follows the commonly used 7.5:1:1.75 ratio.',
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
            Add qualitative result here — T1 / T2 / GT / Prediction grid
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
