// Edit this file to update ablation study results.

export interface AblationRow {
  id: string
  label: string
  description: string
  pre: number
  rec: number
  f1: number
  iou: number
  oa: number
  params: number
  delta_f1?: number
  delta_iou?: number
}

export const ablationRows: AblationRow[] = [
  {
    id: 'A0',
    label: 'A0 – FPN Baseline',
    description: 'SimpleCNN encoder + basic FPN decoder',
    pre: 77.27,
    rec: 76.15,
    f1: 76.63,
    iou: 62.12,
    oa: 83.80,
    params: 7.84,
  },
  {
    id: 'A1',
    label: 'A1 – MambaVision-S + FPN',
    description: 'Replace SimpleCNN with MambaVision-S encoder',
    pre: 93.65,
    rec: 94.09,
    f1: 93.87,
    iou: 88.45,
    oa: 95.72,
    params: 53.54,
    delta_f1: 17.24,
    delta_iou: 26.33,
  },
  {
    id: 'A2',
    label: 'A2 – + D-RBI (unsigned)',
    description: 'Add D-RBI with absolute difference only',
    pre: 92.70,
    rec: 93.98,
    f1: 93.33,
    iou: 87.50,
    oa: 95.35,
    params: 54.98,
    delta_f1: -0.54,
    delta_iou: -0.95,
  },
  {
    id: 'A3',
    label: 'A3 – + Signed Difference',
    description: 'Add signed temporal difference to D-RBI',
    pre: 93.69,
    rec: 94.88,
    f1: 94.28,
    iou: 89.19,
    oa: 95.99,
    params: 55.34,
    delta_f1: 0.95,
    delta_iou: 1.69,
  },
  {
    id: 'A4',
    label: 'A4 – + ARF-FPN',
    description: 'Replace FPN decoder with ARF-FPN',
    pre: 93.94,
    rec: 94.78,
    f1: 94.36,
    iou: 89.32,
    oa: 96.05,
    params: 65.12,
    delta_f1: 0.08,
    delta_iou: 0.13,
  },
  {
    id: 'A5',
    label: 'A5 – + Boundary Residual',
    description: 'Add boundary residual correction head',
    pre: 92.96,
    rec: 94.22,
    f1: 93.59,
    iou: 87.94,
    oa: 95.53,
    params: 65.19,
    delta_f1: -0.77,
    delta_iou: -1.38,
  },
  {
    id: 'A6',
    label: 'A6 – Full Model',
    description: 'Full model with CRAMLite + coarse/boundary auxiliary losses',
    pre: 95.47,
    rec: 95.87,
    f1: 95.67,
    iou: 91.71,
    oa: 96.98,
    params: 65.40,
    delta_f1: 2.08,
    delta_iou: 3.77,
  },
]

export const ablationInsights = [
  {
    from: 'A0',
    to: 'A1',
    delta_f1: 17.24,
    delta_iou: 26.33,
    comment: 'MambaVision encoder is the dominant contributor',
  },
  {
    from: 'A2',
    to: 'A3',
    delta_f1: 0.95,
    delta_iou: 1.69,
    comment: 'Signed difference captures temporal direction',
  },
  {
    from: 'A4',
    to: 'A5',
    delta_f1: -0.77,
    delta_iou: -1.38,
    comment: 'Boundary head alone slightly destabilizes training',
  },
  {
    from: 'A5',
    to: 'A6',
    delta_f1: 2.08,
    delta_iou: 3.77,
    comment: 'Full objective stabilizes and improves boundary refinement',
  },
]
