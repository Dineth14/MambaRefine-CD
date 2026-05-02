import { Github } from 'lucide-react'
import { REPO_URL } from '../data/architecture'

export default function Footer() {
  return (
    <footer className="bg-main-text text-slate-400 py-12 mt-20">
      <div className="max-w-container mx-auto px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
          <div>
            <p className="text-white font-bold text-sm mb-1">MambaRefine-CD</p>
            <p className="text-xs">Binary change detection for remote sensing imagery.</p>
            <p className="text-xs mt-1">All reported numbers are from verified training logs. Do not cite without running the evaluation protocol.</p>
          </div>
          <div className="flex flex-col items-start sm:items-end gap-2">
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-slate-300 hover:text-white transition-colors"
            >
              <Github size={16} />
              GitHub Repository
            </a>
            <p className="text-xs">MambaVision backbone: HAT-Lab/MambaVision (MIT License)</p>
          </div>
        </div>
        <div className="mt-8 pt-6 border-t border-slate-700 text-xs text-slate-500">
          This website presents preliminary research results. Numbers may change with further experimentation.
        </div>
      </div>
    </footer>
  )
}
