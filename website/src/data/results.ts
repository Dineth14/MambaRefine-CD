// Edit this file to update main experiment results.
// All numbers here are from verified training logs. Do not modify unless you have new logs.

export interface ResultRow {
  dataset: string
  setting: string
  pre: number
  rec: number
  f1: number
  iou: number
  oa: number
  params: string
  threshold: number
}

export const mainResults: ResultRow[] = [
  {
    dataset: 'DSIFN-CD',
    setting: 'Full model avg, clean split',
    pre: 95.47,
    rec: 95.87,
    f1: 95.67,
    iou: 91.71,
    oa: 96.98,
    params: '65.40M',
    threshold: 0.60,
  },
  {
    dataset: 'DSIFN-CD',
    setting: 'Best single run, clean split',
    pre: 96.26,
    rec: 96.53,
    f1: 96.40,
    iou: 93.04,
    oa: 97.47,
    params: '65.40M',
    threshold: 0.60,
  },
  {
    dataset: 'WHU-CD',
    setting: 'Full model, standard split',
    pre: 95.58,
    rec: 94.74,
    f1: 95.15,
    iou: 90.76,
    oa: 99.54,
    params: '65.40M',
    threshold: 0.55,
  },
]

export const backboneScaling = [
  {
    label: 'MambaVision-T (tiny)',
    params: 46.73,
    pre: 94.33,
    rec: 94.84,
    f1: 94.58,
    iou: 89.72,
    oa: 96.24,
    note: 'boundary_residual=False in ablation trace',
  },
  {
    label: 'MambaVision-S (small) ★',
    params: 65.40,
    pre: 95.47,
    rec: 95.87,
    f1: 95.67,
    iou: 91.71,
    oa: 96.98,
    note: 'Canonical run – boundary_residual enabled',
  },
  {
    label: 'MambaVision-B (base)',
    params: 113.37,
    pre: 95.66,
    rec: 95.89,
    f1: 95.77,
    iou: 91.89,
    oa: 97.07,
    note: 'boundary_residual=False in ablation trace',
  },
]
