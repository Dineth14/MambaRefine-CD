import { motion } from 'framer-motion'
import { ArrowRight, Github, BarChart2 } from 'lucide-react'
import { REPO_URL } from '../data/architecture'

const metrics = [
  { label: 'DSIFN-CD Avg F1', value: '95.67%', color: 'text-primary', bg: 'bg-soft-blue' },
  { label: 'DSIFN-CD Avg IoU', value: '91.71%', color: 'text-secondary', bg: 'bg-soft-green' },
  { label: 'WHU-CD F1', value: '95.15%', color: 'text-orange-600', bg: 'bg-soft-orange' },
]

const tags = [
  'Binary Change Detection',
  'Remote Sensing',
  'MambaVision',
  'Region-Boundary Refinement',
  'Boundary-Aware Learning',
]

export default function Hero() {
  return (
    <section id="overview" className="pt-28 pb-20 bg-gradient-to-b from-white to-slate-50">
      <div className="max-w-container mx-auto px-4 sm:px-6">
        {/* Eyebrow */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-xs font-semibold tracking-widest text-primary uppercase mb-4"
        >
          Remote Sensing • Change Detection
        </motion.p>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="text-4xl md:text-5xl lg:text-6xl font-bold text-main-text tracking-tight mb-5 max-w-4xl"
        >
          Mamba<span className="text-primary">Refine</span>-CD
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.10 }}
          className="text-lg md:text-xl text-muted-text max-w-3xl mb-6 leading-relaxed"
        >
          A MambaVision-based binary change detection model that explicitly separates{' '}
          <span className="font-semibold text-main-text">region-level change evidence</span> and{' '}
          <span className="font-semibold text-main-text">boundary-level refinement</span> for remote sensing image pairs.
        </motion.p>

        {/* Tags */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="flex flex-wrap gap-2 mb-10"
        >
          {tags.map((t) => (
            <span key={t} className="tag bg-slate-100 text-slate-700 border border-slate-200">
              {t}
            </span>
          ))}
        </motion.div>

        {/* Metric cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10 max-w-2xl"
        >
          {metrics.map((m) => (
            <div key={m.label} className={`card card-hover ${m.bg} border-0 text-center py-6`}>
              <p className={`text-3xl font-bold ${m.color} mb-1`}>{m.value}</p>
              <p className="text-xs text-muted-text font-medium">{m.label}</p>
            </div>
          ))}
        </motion.div>

        {/* Short description */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.25 }}
          className="text-sm text-muted-text max-w-2xl mb-8 leading-relaxed"
        >
          MambaRefine-CD takes two remote sensing images from different times and predicts a binary change map. The model
          uses a shared MambaVision encoder, Differential Region-Boundary Interaction (D-RBI), adaptive receptive field
          decoding (ARF-FPN), and a bounded boundary residual correction.
        </motion.p>

        {/* Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.30 }}
          className="flex flex-wrap gap-3"
        >
          <a href="#architecture" className="btn-primary">
            <BarChart2 size={16} /> View Architecture
          </a>
          <a href="#results" className="btn-secondary">
            <ArrowRight size={16} /> See Results
          </a>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="btn-secondary">
            <Github size={16} /> GitHub Repository
          </a>
        </motion.div>
      </div>
    </section>
  )
}
