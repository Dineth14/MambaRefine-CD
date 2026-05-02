# MambaRefine-CD Website

A React + Vite + TypeScript + Tailwind CSS project website for the MambaRefine-CD change detection research.

## Local Development

```bash
# Install Node.js via nvm if needed
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"
nvm install 20

# Install dependencies
npm install

# Start dev server
npm run dev
# → http://localhost:5173
```

## Build

```bash
npm run build
# Output: dist/
```

## Deploy to GitHub Pages

```bash
npm run deploy
# Pushes dist/ to the gh-pages branch of https://github.com/Dineth14/MambaRefine-CD
# Live at: https://dineth14.github.io/MambaRefine-CD/
```

**First-time setup:** Ensure GitHub Pages is enabled in the repository settings under Settings → Pages → Source: `gh-pages` branch.

## Update Results

All displayed numbers come from TypeScript data files — no hardcoded strings in components:

| File | Purpose |
|---|---|
| `src/data/results.ts` | Main quantitative results (DSIFN-CD, WHU-CD) |
| `src/data/ablations.ts` | Ablation study rows and delta insights |
| `src/data/comparisons.ts` | Literature comparison tables |
| `src/data/architecture.ts` | Architecture module descriptions and model stats |

## Stack

- React 18 + TypeScript 5
- Vite 5
- Tailwind CSS 3
- Framer Motion 11 (scroll animations)
- Recharts 2 (bar/scatter charts)
- react-katex + KaTeX (math equations)
- gh-pages (deployment)
