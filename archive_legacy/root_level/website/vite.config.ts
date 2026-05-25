import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// IMPORTANT: Change '/MambaRefine-CD/' to match your GitHub repository name.
// e.g. if your repo is https://github.com/Dineth14/my-project, set base: '/my-project/'
const base = process.env.VITE_BASE_PATH || '/MambaRefine-CD/'

export default defineConfig({
  plugins: [react()],
  base,
})
