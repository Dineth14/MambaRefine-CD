interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  sublabel?: string
  color?: string
  bg?: string
}

export default function MetricCard({ label, value, unit = '', sublabel, color = 'text-primary', bg = 'bg-soft-blue' }: MetricCardProps) {
  return (
    <div className={`card card-hover ${bg} border-0 text-center py-6 px-4`}>
      <p className={`text-3xl font-bold ${color} mb-1`}>
        {value}{unit}
      </p>
      <p className="text-xs font-semibold text-muted-text uppercase tracking-wide">{label}</p>
      {sublabel && <p className="text-xs text-slate-400 mt-1">{sublabel}</p>}
    </div>
  )
}
