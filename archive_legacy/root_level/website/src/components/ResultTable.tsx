import { mainResults } from '../data/results'

export default function ResultTable() {
  return (
    <div className="overflow-x-auto rounded-xl border border-border shadow-card">
      <table className="result-table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Setting</th>
            <th>Pre (%)</th>
            <th>Rec (%)</th>
            <th>F1 (%)</th>
            <th>IoU (%)</th>
            <th>OA (%)</th>
            <th>Params</th>
            <th>Threshold</th>
          </tr>
        </thead>
        <tbody>
          {mainResults.map((r, i) => (
            <tr key={i}>
              <td className="font-medium">{r.dataset}</td>
              <td className="text-muted-text text-xs max-w-[200px]">{r.setting}</td>
              <td>{r.pre.toFixed(2)}</td>
              <td>{r.rec.toFixed(2)}</td>
              <td className="font-semibold text-primary">{r.f1.toFixed(2)}</td>
              <td className="font-semibold">{r.iou.toFixed(2)}</td>
              <td>{r.oa.toFixed(2)}</td>
              <td className="font-mono text-xs">{r.params}</td>
              <td className="font-mono text-xs">{r.threshold}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
