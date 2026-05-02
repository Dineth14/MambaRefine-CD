import Section from '../components/Section'
import Hero from '../components/Hero'
import ArchitectureDiagram from '../components/ArchitectureDiagram'
import DRBIDiagram from '../components/DRBIDiagram'
import LossDiagram from '../components/LossDiagram'
import MathBlock from '../components/MathBlock'
import DatasetCard from '../components/DatasetCard'
import ResultTable from '../components/ResultTable'
import AblationChart from '../components/AblationChart'
import ComparisonChart from '../components/ComparisonChart'
import RankingTable from '../components/RankingTable'
import Timeline from '../components/Timeline'
import Footer from '../components/Footer'
import MetricCard from '../components/MetricCard'
import { ablationInsights } from '../data/ablations'

export default function Home() {
  return (
    <>
      <Hero />

      {/* Motivation */}
      <Section
        id="motivation"
        eyebrow="Why this work"
        title="Motivation"
        subtitle="Existing change detection models often produce coarse boundaries or rely on generic feature subtraction that discards temporal direction."
        className="bg-white"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              title: 'Temporal Direction Loss',
              body: 'Absolute difference |F_T2 – F_T1| discards the direction of change. We retain both signed and unsigned differences to encode temporal direction explicitly.',
              icon: '⟷',
              color: 'text-primary',
            },
            {
              title: 'Boundary Imprecision',
              body: 'Segmentation decoders trained only on region-level binary cross-entropy often produce coarse object boundaries. A dedicated boundary residual head addresses this.',
              icon: '⬛',
              color: 'text-secondary',
            },
            {
              title: 'Scale Sensitivity',
              body: 'Changed regions span multiple scales — small rooftops to large land parcels. Adaptive Receptive Field FPN (ARF-FPN) adapts kernel coverage per scale level.',
              icon: '⊞',
              color: 'text-orange-600',
            },
          ].map((m) => (
            <div key={m.title} className="card card-hover">
              <div className={`text-2xl font-bold ${m.color} mb-2`}>{m.icon}</div>
              <h3 className="font-bold text-main-text text-sm mb-2">{m.title}</h3>
              <p className="text-xs text-muted-text leading-relaxed">{m.body}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Architecture */}
      <Section
        id="architecture"
        eyebrow="Design"
        title="Architecture"
        subtitle="MambaRefine-CD processes paired remote sensing images through a shared backbone, a differential interaction module, an adaptive decoder, and a boundary refinement head."
        className="bg-slate-50"
      >
        <ArchitectureDiagram />
      </Section>

      {/* Math */}
      <Section
        id="math"
        eyebrow="Formulation"
        title="Mathematical Formulation"
        subtitle="Core equations that define D-RBI, ARF-FPN, and the training objective."
        className="bg-white"
      >
        <div className="space-y-8 max-w-3xl">
          <div className="card">
            <h3 className="font-bold text-sm text-main-text mb-1">Differential Region-Boundary Interaction (D-RBI)</h3>
            <p className="text-xs text-muted-text mb-3">
              For encoder features <span className="font-mono">F1</span> and <span className="font-mono">F2</span> at the same scale:
            </p>
            <MathBlock tex="\Delta^+ = |F_2 - F_1|, \quad \Delta^s = F_2 - F_1" />
            <MathBlock tex="\tilde{D} = \sigma\!\left(g\!\left(\Delta^+, \Delta^s, [F_1; F_2]\right)\right) \odot \phi_r(\Delta^+) + \phi_b(\Delta^s)" />
            <p className="text-xs text-muted-text mt-2">
              where <span className="font-mono">φ_r</span> is the region branch (3×3 conv + GELU + BN), <span className="font-mono">φ_b</span> is the boundary branch (Laplacian-guided), and <span className="font-mono">g</span> is the fusion gate.
            </p>
          </div>

          <div className="card">
            <h3 className="font-bold text-sm text-main-text mb-1">Boundary Residual Correction</h3>
            <p className="text-xs text-muted-text mb-3">
              The boundary head predicts a residual <span className="font-mono">r</span> added to the coarse logits:
            </p>
            <MathBlock tex="\hat{y} = \sigma\!\left(y_{\text{coarse}} + \lambda_r \cdot r_\theta(D_l)\right)" />
            <p className="text-xs text-muted-text mt-2">
              <span className="font-mono">D_l</span> is the lowest-scale D-RBI feature. <span className="font-mono">λ_r</span> is the boundary residual weight (default 1.0).
            </p>
          </div>

          <div className="card">
            <h3 className="font-bold text-sm text-main-text mb-1">Training Objective</h3>
            <MathBlock tex="\mathcal{L} = \mathcal{L}_{\text{bce}} + \mathcal{L}_{\text{dice}} + \lambda_f \mathcal{L}_{\text{focal}} + \lambda_b \mathcal{L}_{\text{boundary}} + \lambda_c \mathcal{L}_{\text{coarse}}" />
            <p className="text-xs text-muted-text mt-2">
              Default weights: λ_f = 0.5, λ_b = 0.3, λ_c = 0.4. Boundary auxiliary loss is active during training only.
            </p>
          </div>

          <div className="card">
            <h3 className="font-bold text-sm text-main-text mb-1">Evaluation Metrics</h3>
            <MathBlock tex="\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{IoU} = \frac{TP}{TP + FP + FN}" />
            <p className="text-xs text-muted-text mt-2">
              Threshold is optimized per-dataset on the validation set by sweeping [0.30, 0.75] in steps of 0.05. Reported threshold applied at test time.
            </p>
          </div>

          <DRBIDiagram />
          <LossDiagram />
        </div>
      </Section>

      {/* Datasets */}
      <Section
        id="datasets"
        eyebrow="Evaluation"
        title="Datasets"
        subtitle="We evaluate on DSIFN-CD and WHU-CD with verified clean splits and reproducible tiling protocols."
        className="bg-slate-50"
      >
        <DatasetCard />
      </Section>

      {/* Results */}
      <Section
        id="results"
        eyebrow="Main Results"
        title="Quantitative Results"
        subtitle="Numbers are from verified training runs. Threshold was swept on the validation set; the chosen value is applied at test time."
        className="bg-white"
      >
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <MetricCard label="DSIFN Avg F1" value="95.67" unit="%" color="text-primary" bg="bg-soft-blue" sublabel="3-run avg" />
          <MetricCard label="DSIFN Avg IoU" value="91.71" unit="%" color="text-secondary" bg="bg-soft-green" sublabel="3-run avg" />
          <MetricCard label="WHU-CD F1" value="95.15" unit="%" color="text-orange-600" bg="bg-soft-orange" sublabel="standard split" />
          <MetricCard label="WHU-CD IoU" value="90.76" unit="%" color="text-purple-600" bg="bg-purple-50" sublabel="standard split" />
        </div>
        <ResultTable />
        <p className="text-xs text-muted-text mt-3">
          Avg = mean over 3 independent training runs. Best = highest-F1 single run. Results are not cherry-picked from hyperparameter ablations.
        </p>
      </Section>

      {/* Ablation */}
      <Section
        id="ablation"
        eyebrow="Component Analysis"
        title="Ablation Study"
        subtitle="Incremental study on DSIFN-CD clean split. Each step adds exactly one component to the previous configuration."
        className="bg-slate-50"
      >
        <AblationChart />

        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {ablationInsights.map((ins) => (
            <div key={ins.from + ins.to} className={`card border ${ins.delta_f1 > 0 ? 'border-green-200 bg-green-50' : 'border-rose-200 bg-rose-50'}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-xs font-bold text-muted-text">{ins.from} → {ins.to}</span>
                <span className={`text-xs font-bold ${ins.delta_f1 > 0 ? 'text-green-700' : 'text-rose-700'}`}>
                  {ins.delta_f1 > 0 ? '+' : ''}{ins.delta_f1.toFixed(2)} F1
                </span>
              </div>
              <p className="text-xs text-muted-text">{ins.comment}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Scaling */}
      <Section
        id="scaling"
        eyebrow="Backbone Scaling"
        title="Backbone Size vs Performance"
        subtitle="Scaling from MambaVision-T to MambaVision-B. Results on DSIFN-CD clean split."
        className="bg-white"
      >
        <ComparisonChart />
        <p className="text-xs text-muted-text mt-3">
          ★ MambaVision-S is the canonical model. Tiny and Base runs used <code>boundary_residual=False</code> in the ablation trace and are not directly comparable to the canonical small run.
        </p>
      </Section>

      {/* Development timeline */}
      <Section
        id="insights"
        eyebrow="Development"
        title="Component Build-Up"
        subtitle="Progressive addition of components. Each step builds on the previous verified configuration."
        className="bg-slate-50"
      >
        <Timeline />
      </Section>

      {/* Comparison */}
      <Section
        id="comparison"
        eyebrow="Literature Comparison"
        title="Comparison with Prior Work"
        subtitle="WHU-CD comparison uses the same standard split. DSIFN-CD is contextual — different papers use different patch protocols."
        className="bg-white"
      >
        <RankingTable />
      </Section>

      {/* Limitations */}
      <Section
        id="limitations"
        eyebrow="Honest Assessment"
        title="Limitations"
        className="bg-slate-50"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-3xl">
          {[
            {
              title: 'DSIFN Protocol Fragmentation',
              body: 'The DSIFN-CD benchmark lacks a universally adopted split. Different papers report results on incompatible patch protocols, making direct comparison unreliable without matching the exact protocol.',
            },
            {
              title: 'Single Dataset per Domain',
              body: 'We evaluate on one urban-change and one building-change dataset. Generalization to other change types (vegetation, flooding, seasonal) is not verified.',
            },
            {
              title: 'Boundary Residual Instability',
              body: 'Adding the boundary residual head alone (A5) slightly reduced performance. It only helps when combined with CRAMLite and full auxiliary losses (A6), suggesting sensitivity to the full objective.',
            },
            {
              title: 'Threshold Sensitivity',
              body: 'Optimal thresholds differ per dataset (0.60 for DSIFN-CD, 0.55 for WHU-CD). A data-dependent threshold strategy is not ideal for deployment without calibration data.',
            },
          ].map((l) => (
            <div key={l.title} className="card border border-amber-200 bg-amber-50">
              <h4 className="font-bold text-sm text-amber-900 mb-1">{l.title}</h4>
              <p className="text-xs text-amber-800 leading-relaxed">{l.body}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Contact / Repo */}
      <Section
        id="contact"
        eyebrow="Repository"
        title="Code & Reproducibility"
        className="bg-white"
      >
        <div className="max-w-2xl">
          <p className="text-sm text-muted-text mb-6 leading-relaxed">
            Training configs, dataset split generation scripts, and evaluation code are available in the repository.
            To reproduce the main DSIFN-CD result, run the <code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">dsifn_full.yaml</code> config on the clean split.
          </p>
          <div className="card bg-slate-50 border border-slate-200 font-mono text-xs text-slate-700 space-y-1">
            <p><span className="text-slate-400"># Clone and install</span></p>
            <p>git clone https://github.com/Dineth14/MambaRefine-CD.git</p>
            <p>cd MambaRefine-CD && pip install -r requirements.txt</p>
            <p className="pt-1"><span className="text-slate-400"># Train on DSIFN-CD</span></p>
            <p>python scripts/train.py --config configs/experiments/dsifn_full.yaml</p>
            <p className="pt-1"><span className="text-slate-400"># Evaluate</span></p>
            <p>python scripts/test.py --config configs/experiments/dsifn_full.yaml --ckpt PATH_TO_BEST</p>
          </div>
        </div>
      </Section>

      <Footer />
    </>
  )
}
