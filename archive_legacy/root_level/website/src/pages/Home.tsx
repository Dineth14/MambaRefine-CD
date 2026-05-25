import { AlertTriangle, Brain, Braces, Database, GitCompare, Layers3, MoveHorizontal, Network, Route, ScanLine, Target, TrendingUp } from 'lucide-react'
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
import Footer from '../components/Footer'
import MetricCard from '../components/MetricCard'
import { ablationInsights } from '../data/ablations'
import { REPO_URL } from '../data/architecture'

const motivationChallenges = [
  {
    title: 'Long-range context',
    body: 'Changed objects can be spatially separated from the contextual evidence needed to recognize them. Mamba-style sequence modeling is attractive because it can model long-range dependencies efficiently.',
    Icon: Network,
    tone: 'text-primary bg-soft-blue',
  },
  {
    title: 'Temporal difference modeling',
    body: 'Illumination, seasonal variation, shadows, and background changes can produce false alarms. The model needs explicit temporal evidence rather than only independent feature decoding.',
    Icon: MoveHorizontal,
    tone: 'text-secondary bg-soft-green',
  },
  {
    title: 'Boundary precision',
    body: 'Changed regions are often small, fragmented, or irregular. Region-level evidence and boundary-level correction are related, but they should not be forced into a single stream.',
    Icon: ScanLine,
    tone: 'text-accent bg-soft-orange',
  },
]

const trainingProtocol = [
  ['Optimizer', 'AdamW'],
  ['LR', '5e-5'],
  ['LR schedule', 'cosine decay'],
  ['Warmup', '2500 iterations'],
  ['Weight decay', '0.01'],
  ['Batch size', '8'],
  ['Mixed precision', 'AMP'],
  ['EMA decay', '0.999'],
  ['Validation', 'every 5000 iterations'],
  ['Threshold sweep', '[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]'],
  ['Test threshold', 'best validation threshold'],
]

const resultInsights = [
  {
    title: 'MambaVision is doing the heavy lifting.',
    evidence: 'A0 to A1 gives +17.56 F1 and +26.44 IoU.',
    Icon: Brain,
  },
  {
    title: 'Temporal direction helps.',
    evidence: 'Adding signed difference improves F1 and IoU over unsigned D-RBI.',
    Icon: GitCompare,
  },
  {
    title: 'Boundary refinement needs stable supervision.',
    evidence: 'Boundary residual alone drops slightly, but the full model improves when CRAMLite, coarse loss, and boundary loss are included.',
    Icon: ScanLine,
  },
  {
    title: 'Bigger backbone is not always worth it.',
    evidence: 'MambaVision-B gives only +0.10 F1 over MambaVision-S for a much larger parameter count.',
    Icon: TrendingUp,
  },
]

const limitations = [
  'DSIFN comparison with literature needs same-protocol evaluation for strict ranking.',
  'Only one canonical WHU full run is currently available.',
  'Boundary residual behavior needs more qualitative visualization.',
  'Cross-dataset generalization should be tested.',
  'Official weights of recent baselines would allow stronger same-split comparison.',
]

const nextSteps = [
  'Train/evaluate on the DSIFN literature patch protocol.',
  'Add qualitative examples with TP/FP/FN/TN color maps.',
  'Evaluate official baseline checkpoints on the same split.',
  'Add boundary-specific metrics if logs are available.',
  'Prepare MERCon paper figures and tables.',
]

function PlaceholderCard({ title }: { title: string }) {
  return (
    <div className="card border-dashed border-slate-300 bg-slate-50 flex min-h-[150px] items-center justify-center text-center">
      <div>
        <p className="text-sm font-semibold text-slate-600">{title}</p>
        <p className="mt-1 text-xs text-slate-400">Add qualitative result here</p>
      </div>
    </div>
  )
}

export default function Home() {
  return (
    <>
      <Hero />

      <Section
        id="motivation"
        eyebrow="Motivation"
        title="Why another change detection model?"
        subtitle="A strong encoder helps, but remote sensing change detection still depends on how temporal evidence and object boundaries are represented."
        className="bg-white"
      >
        <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-6">
          <div className="card">
            <p className="text-sm leading-relaxed text-muted-text">
              Remote sensing change detection is difficult because changed regions can be small, fragmented, or irregular. Illumination, seasonal variation, shadows, and background changes can create false alarms. CNNs capture local detail but may miss long-range context, while transformer-based models capture global context but can be expensive.
            </p>
            <p className="mt-4 text-sm leading-relaxed text-muted-text">
              Mamba-style models are attractive because they model long-range dependencies efficiently. However, using a strong encoder alone is not enough. Boundary quality and temporal feature construction still matter, especially when a binary map must be spatially precise.
            </p>
          </div>
          <div className="card border-primary/30 bg-blue-50">
            <p className="text-xs font-semibold uppercase tracking-wide text-primary">Main hypothesis</p>
            <p className="mt-3 text-xl font-bold leading-snug text-main-text">
              Region-level change and boundary-level change should be handled as related but separate feature streams.
            </p>
          </div>
        </div>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {motivationChallenges.map(({ title, body, Icon, tone }) => (
            <div key={title} className="card card-hover">
              <div className={`mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg ${tone}`}>
                <Icon size={20} />
              </div>
              <h3 className="text-base font-bold text-main-text">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-text">{body}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section
        id="architecture"
        eyebrow="Architecture"
        title="Region-boundary temporal refinement"
        subtitle="The model uses a shared MambaVision encoder, D-RBI temporal evidence, CRAMLite region attention, ARF-FPN decoding, and a bounded boundary residual."
        className="bg-slate-50"
      >
        <ArchitectureDiagram />
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <DRBIDiagram />
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-text">Boundary residual</h3>
              <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-muted-text">
                Coarse map + Sobel uncertainty + boundary feature -&gt; delta -&gt; bounded correction.
              </div>
              <p className="mt-3 text-sm leading-relaxed text-muted-text">
                The final logits are computed as <span className="font-mono">P_f = P_c + 0.1 tanh(delta)</span>, so the boundary head can refine uncertain edges but cannot fully override the coarse prediction.
              </p>
            </div>
            <LossDiagram />
          </div>
        </div>
      </Section>

      <Section
        id="math"
        eyebrow="Mathematical Intuition"
        title="How the model constructs and refines change evidence"
        subtitle="The equations below are intended to explain the design rather than replace the implementation details."
        className="bg-white"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="font-bold text-main-text">4.1 Shared Encoding</h3>
            <MathBlock tex="F_1^s = E(I_1), \quad F_2^s = E(I_2)" />
            <p className="text-sm leading-relaxed text-muted-text">The same encoder is used for both dates, so the feature spaces remain aligned. This makes temporal comparison more stable.</p>
          </div>
          <div className="card">
            <h3 className="font-bold text-main-text">4.2 Temporal Evidence Construction</h3>
            <MathBlock tex="T^s = [F_1^s, F_2^s, |F_2^s - F_1^s|, F_2^s - F_1^s]" />
            <p className="text-sm leading-relaxed text-muted-text">Raw features preserve appearance at each time. Absolute difference captures change magnitude. Signed difference preserves temporal direction.</p>
          </div>
          <div className="card">
            <h3 className="font-bold text-main-text">4.3 D-RBI Region and Boundary Streams</h3>
            <MathBlock tex="D^s = \phi(T^s)" />
            <MathBlock tex="G_r^s = a_r + (b_r-a_r)\sigma(\psi_r(D^s)), \quad R^s = G_r^s \odot D^s" />
            <MathBlock tex="G_b^s = a_b + (b_b-a_b)\sigma(\psi_b(\operatorname{Sobel}(D^s))), \quad B^s = G_b^s \odot D^s" />
            <p className="text-sm leading-relaxed text-muted-text">The region gate uses bounds [0.2, 0.8]. The boundary gate uses bounds [0.0, 0.4] and is deliberately limited to avoid unstable boundary over-amplification.</p>
          </div>
          <div className="card">
            <h3 className="font-bold text-main-text">4.4 CRAMLite</h3>
            <MathBlock tex="A^s = \sigma(f_{\text{spatial}}(R^s))" />
            <MathBlock tex="\tilde{R}^s = R^s \odot (1 + \alpha A^s)" />
            <p className="text-sm leading-relaxed text-muted-text">CRAMLite is residual spatial attention. The multiplier starts close to the original region feature and can learn to enhance likely change regions.</p>
          </div>
          <div className="card">
            <h3 className="font-bold text-main-text">4.5 Adaptive Receptive Field FPN</h3>
            <MathBlock tex="Y_r^s = \sum_{r \in \{1,2,4,8\}} w_r \cdot \operatorname{Conv}_{3 \times 3}^{(d=r)}(R^s)" />
            <p className="text-sm leading-relaxed text-muted-text">Small buildings need local detail, while larger urban changes need wider context. ARF-FPN learns per-image weights over dilation branches.</p>
          </div>
          <div className="card">
            <h3 className="font-bold text-main-text">4.6 Boundary Residual Refinement</h3>
            <MathBlock tex="E = \operatorname{Sobel}(\sigma(P_c))" />
            <MathBlock tex="\delta = h([B_0, P_c, E])" />
            <MathBlock tex="P_f = P_c + 0.1 \tanh(\delta)" />
            <p className="text-sm leading-relaxed text-muted-text">The model predicts a coarse map first, then applies a small bounded correction around boundary-uncertain regions.</p>
          </div>
          <div className="card lg:col-span-2">
            <h3 className="font-bold text-main-text">4.7 Loss Function</h3>
            <MathBlock tex="\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice} + 0.4\mathcal{L}_{coarse} + 0.1\mathcal{L}_{boundary}" />
            <p className="text-sm leading-relaxed text-muted-text">BCE gives pixel-level classification pressure. Dice helps with class imbalance. Coarse loss stabilizes the ARF-FPN prediction. Boundary loss encourages edge-aware refinement.</p>
          </div>
        </div>
      </Section>

      <Section
        id="datasets"
        eyebrow="Datasets and Protocol"
        title="Verified evaluation protocol"
        subtitle="The tables report only the supplied split and training details. Test thresholds are selected from validation, not tuned on the test set."
        className="bg-slate-50"
      >
        <DatasetCard />
        <div className="card mt-6">
          <div className="mb-4 flex items-center gap-2">
            <Database size={18} className="text-primary" />
            <h3 className="font-bold text-main-text">Training protocol</h3>
          </div>
          <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {trainingProtocol.map(([key, value]) => (
              <div key={key} className="rounded-lg border border-border bg-white p-3">
                <dt className="text-xs font-semibold text-muted-text">{key}</dt>
                <dd className="mt-1 text-sm font-medium text-main-text">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </Section>

      <Section
        id="results"
        eyebrow="Main Results"
        title="Quantitative results"
        subtitle="Numbers are from verified runs. No state-of-the-art claim is made because comparison protocols are not always identical."
        className="bg-white"
      >
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <MetricCard label="DSIFN Avg F1" value="95.67" unit="%" color="text-primary" bg="bg-soft-blue" sublabel="full model avg" />
          <MetricCard label="DSIFN Best F1" value="96.40" unit="%" color="text-primary" bg="bg-blue-50" sublabel="single run" />
          <MetricCard label="WHU F1" value="95.15" unit="%" color="text-accent" bg="bg-soft-orange" sublabel="standard split" />
          <MetricCard label="DSIFN Avg IoU" value="91.71" unit="%" color="text-secondary" bg="bg-soft-green" sublabel="full model avg" />
          <MetricCard label="WHU IoU" value="90.76" unit="%" color="text-secondary" bg="bg-emerald-50" sublabel="standard split" />
        </div>
        <ResultTable />
        <p className="mt-4 max-w-4xl text-sm leading-relaxed text-muted-text">
          The results show that the model gives strong changed-class F1 and IoU on both datasets. On DSIFN-CD, the average of two canonical full-model runs reaches 95.67% F1, while the best single run reaches 96.40% F1. On WHU-CD, the model reaches 95.15% F1 under the standard split.
        </p>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <PlaceholderCard title="DSIFN qualitative panel" />
          <PlaceholderCard title="WHU qualitative panel" />
        </div>
      </Section>

      <Section
        id="ablation"
        eyebrow="Ablation Study"
        title="Which components matter?"
        subtitle="The DSIFN-CD clean-split ablation isolates the contribution of the encoder, temporal evidence, adaptive decoding, and boundary refinement."
        className="bg-slate-50"
      >
        <AblationChart />
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {ablationInsights.map((ins) => (
            <div key={ins.from + ins.to} className={`card border ${ins.delta_f1 > 0 ? 'border-green-200 bg-green-50' : 'border-orange-200 bg-orange-50'}`}>
              <p className="font-mono text-xs font-bold text-muted-text">{ins.from} -&gt; {ins.to}</p>
              <p className={`mt-2 text-lg font-bold ${ins.delta_f1 > 0 ? 'text-green-700' : 'text-orange-700'}`}>
                {ins.delta_f1 > 0 ? '+' : ''}{ins.delta_f1.toFixed(2)} F1, {ins.delta_iou > 0 ? '+' : ''}{ins.delta_iou.toFixed(2)} IoU
              </p>
              <p className="mt-2 text-sm leading-relaxed text-muted-text">{ins.comment}</p>
            </div>
          ))}
        </div>
        <div className="card mt-6">
          <p className="text-sm leading-relaxed text-muted-text">
            The largest gain comes from replacing the SimpleCNN baseline with MambaVision-S. Signed temporal difference improves D-RBI, and ARF-FPN gives a small positive gain. Boundary residual alone slightly decreases performance. The full configuration recovers and improves boundary refinement when combined with CRAMLite and the auxiliary loss schedule, suggesting that boundary refinement must be trained as part of a stabilized full objective, not simply attached as an isolated head.
          </p>
        </div>
      </Section>

      <Section
        id="scaling"
        eyebrow="Backbone Scaling"
        title="Efficiency-accuracy trade-off"
        subtitle="MambaVision-B improves only slightly over MambaVision-S despite much higher parameter count."
        className="bg-white"
      >
        <ComparisonChart />
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <p>The tiny and base scaling runs had <code>boundary_residual_enabled=False</code> in the ablation traces, while the small canonical run had boundary residual enabled.</p>
        </div>
      </Section>

      <Section
        id="comparison"
        eyebrow="Comparison"
        title="Comparison with recent methods"
        subtitle="WHU-CD is ranked directly from the supplied table. DSIFN-CD is shown as contextual evidence because literature protocols can differ."
        className="bg-slate-50"
      >
        <RankingTable />
        <div className="card mt-6">
          <p className="text-sm leading-relaxed text-muted-text">
            On WHU-CD, MambaRefine-CD closely matches Mamba-CD and obtains higher recall, with only a small gap in F1 and IoU. On DSIFN-CD, MambaRefine-CD gives strong performance under the verified clean split, but direct ranking against literature DSIFN values requires the same patch-level protocol.
          </p>
        </div>
      </Section>

      <Section
        id="learned"
        eyebrow="What We Learned"
        title="What the results say about the model"
        subtitle="The ablations suggest where the model is strong and where the design needs more evidence."
        className="bg-white"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {resultInsights.map(({ title, evidence, Icon }) => (
            <div key={title} className="card card-hover">
              <Icon size={22} className="mb-4 text-primary" />
              <h3 className="text-lg font-bold text-main-text">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-text">{evidence}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section
        id="limitations"
        eyebrow="Limitations and Next Steps"
        title="What still needs to be validated"
        subtitle="These points keep the interpretation tied to the verified experiments rather than overclaiming."
        className="bg-slate-50"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <div className="mb-4 flex items-center gap-2">
              <Target size={18} className="text-accent" />
              <h3 className="font-bold text-main-text">Limitations</h3>
            </div>
            <ul className="space-y-3 text-sm leading-relaxed text-muted-text">
              {limitations.map((item) => <li key={item}>- {item}</li>)}
            </ul>
          </div>
          <div className="card">
            <div className="mb-4 flex items-center gap-2">
              <Route size={18} className="text-secondary" />
              <h3 className="font-bold text-main-text">Next steps</h3>
            </div>
            <ul className="space-y-3 text-sm leading-relaxed text-muted-text">
              {nextSteps.map((item) => <li key={item}>- {item}</li>)}
            </ul>
          </div>
        </div>
      </Section>

      <Section
        id="contact"
        eyebrow="Repository / Contact"
        title="Code and reproducibility"
        subtitle="The website is static and deployable to GitHub Pages. All displayed numbers live in local TypeScript data files."
        className="bg-white"
      >
        <div className="grid grid-cols-1 lg:grid-cols-[0.9fr_1.1fr] gap-6">
          <div className="card">
            <Layers3 size={22} className="mb-4 text-primary" />
            <p className="text-sm leading-relaxed text-muted-text">
              Edit the repository link in <code>src/data/architecture.ts</code>. Results, ablations, and comparison tables are separated into data files so new verified logs can be added without changing presentation components.
            </p>
            <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="btn-primary mt-5">
              Open GitHub Repository
            </a>
          </div>
          <div className="card bg-slate-50">
            <div className="mb-4 flex items-center gap-2">
              <Braces size={18} className="text-primary" />
              <h3 className="font-bold text-main-text">Website commands</h3>
            </div>
            <pre className="overflow-x-auto rounded-lg bg-white p-4 text-xs text-slate-700 border border-border">{`npm install
npm run dev
npm run build
npm run deploy`}</pre>
          </div>
        </div>
      </Section>

      <Footer />
    </>
  )
}
