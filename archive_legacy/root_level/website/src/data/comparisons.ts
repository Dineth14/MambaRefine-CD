// Edit this file to update comparison tables.
// WHU-CD: direct ranking comparison is valid (same standard split).
// DSIFN-CD: contextual only — different papers may use different patch protocols.

export interface ComparisonRow {
  method: string
  year: number
  pre: number | null
  rec: number | null
  f1: number
  iou: number
  oa: number | null
  isOurs?: boolean
  rank?: 1 | 2 | 3
  note?: string
}

export const whuComparison: ComparisonRow[] = [
  {
    method: 'ChangeFormer',
    year: 2022,
    pre: 91.83,
    rec: 88.02,
    f1: 89.88,
    iou: 81.63,
    oa: 99.12,
  },
  {
    method: 'RSM-CD',
    year: 2023,
    pre: 93.37,
    rec: 90.42,
    f1: 91.87,
    iou: 84.96,
    oa: null,
  },
  {
    method: 'CDMamba',
    year: 2024,
    pre: 95.58,
    rec: 92.01,
    f1: 93.76,
    iou: 88.26,
    oa: 99.51,
  },
  {
    method: 'BiFA',
    year: 2024,
    pre: 95.15,
    rec: 93.60,
    f1: 94.37,
    iou: 89.34,
    oa: 99.56,
  },
  {
    method: 'MambaRefine-CD',
    year: 2025,
    pre: 95.58,
    rec: 94.74,
    f1: 95.15,
    iou: 90.76,
    oa: 99.54,
    isOurs: true,
    rank: 2,
  },
  {
    method: 'Mamba-CD',
    year: 2026,
    pre: 96.52,
    rec: 93.91,
    f1: 95.20,
    iou: 90.83,
    oa: 99.62,
    note: 'Peng et al., JSTARS 2026',
    rank: 1,
  },
]

// Sorted by F1 descending for display
export const whuSorted = [...whuComparison].sort((a, b) => b.f1 - a.f1)

export const dsifnComparison: ComparisonRow[] = [
  {
    method: 'ADSFNet',
    year: 2023,
    pre: 94.79,
    rec: 95.24,
    f1: 95.01,
    iou: 90.50,
    oa: 98.30,
    note: 'Literature patch-level protocol',
  },
  {
    method: 'Mamba-CD',
    year: 2026,
    pre: 95.60,
    rec: 95.61,
    f1: 95.61,
    iou: 91.69,
    oa: 98.51,
    note: 'Peng et al., JSTARS 2026 – patch protocol',
  },
  {
    method: 'MambaRefine-CD (avg)',
    year: 2025,
    pre: 95.47,
    rec: 95.87,
    f1: 95.67,
    iou: 91.71,
    oa: 96.98,
    isOurs: true,
    note: 'Our clean split (2758/394/789 images, 0.25 overlap tiles)',
  },
  {
    method: 'MambaRefine-CD (best)',
    year: 2025,
    pre: 96.26,
    rec: 96.53,
    f1: 96.40,
    iou: 93.04,
    oa: 97.47,
    isOurs: true,
    note: 'Our clean split – single best run',
  },
]
