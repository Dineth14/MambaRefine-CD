# MambaRefine-CD Website

A polished static project website for **MambaRefine-CD: Region-Boundary Temporal Refinement with MambaVision for Remote Sensing Change Detection**.

The site is built as a research presentation for GitHub Pages. It has no backend; all architecture descriptions, result tables, ablation values, and comparison data are stored locally in TypeScript files.

## Tech Stack

- React + Vite + TypeScript
- Tailwind CSS
- Framer Motion
- Recharts
- Lucide React
- KaTeX via `react-katex`
- `gh-pages` for deployment

## Local Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
# Output: dist/
```

## Deploy to GitHub Pages

```bash
npm run deploy
```

This builds `dist/` and publishes it to the `gh-pages` branch.

First-time setup:

1. Confirm the repository URL in `src/data/architecture.ts`.
2. Confirm the GitHub Pages base path in `vite.config.ts`.
3. Enable GitHub Pages in repository settings: Settings -> Pages -> Source: `gh-pages` branch.

If your repository name is not `MambaRefine-CD`, set the base path before building:

```bash
VITE_BASE_PATH=/REPOSITORY_NAME/ npm run build
```

## Update Results

All displayed numbers come from TypeScript data files. Update them only from verified logs.

| File | Purpose |
|---|---|
| `src/data/results.ts` | Main quantitative results (DSIFN-CD, WHU-CD) |
| `src/data/ablations.ts` | Ablation study rows and delta insights |
| `src/data/comparisons.ts` | Literature comparison tables |
| `src/data/architecture.ts` | Architecture module descriptions and model stats |
