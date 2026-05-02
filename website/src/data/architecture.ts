// Edit this file to update architecture metadata and GitHub repository URL.

export const REPO_URL = 'https://github.com/Dineth14/MambaRefine-CD'

export const architectureModules = [
  {
    id: 'encoder',
    name: 'Shared MambaVision Encoder',
    short: 'Encoder',
    description:
      'MambaVision-S (small) processes both the pre-change and post-change images with shared weights, producing 4 aligned multi-scale feature maps at 1/4, 1/8, 1/16, 1/32 of the input resolution.',
    channels: [96, 192, 384, 768],
    color: 'bg-blue-50 border-blue-200 text-blue-900',
    accent: '#2563eb',
  },
  {
    id: 'drbi',
    name: 'D-RBI',
    short: 'D-RBI',
    description:
      'Differential Region-Boundary Interaction constructs temporal evidence from raw features, absolute difference, and signed difference. A compression step produces a unified feature, which is split into region and boundary streams via learned gating.',
    color: 'bg-teal-50 border-teal-200 text-teal-900',
    accent: '#0f766e',
  },
  {
    id: 'cram',
    name: 'CRAMLite',
    short: 'CRAMLite',
    description:
      'A lightweight residual spatial attention module applied to the region stream. Uses a (1 + αA) multiplicative gate so initial behavior is close to the identity; attention learns to enhance change-likely regions.',
    color: 'bg-purple-50 border-purple-200 text-purple-900',
    accent: '#7c3aed',
  },
  {
    id: 'arf',
    name: 'ARF-FPN',
    short: 'ARF-FPN',
    description:
      'Adaptive Receptive Field FPN uses parallel dilated convolutions (d=1,2,4,8) at each scale. Learned per-image weights blend the branches, giving flexibility for both small and large changed objects.',
    color: 'bg-orange-50 border-orange-200 text-orange-900',
    accent: '#f97316',
  },
  {
    id: 'boundary',
    name: 'Boundary Residual',
    short: 'Bnd-Residual',
    description:
      'The coarse map P_c is corrected by a small bounded delta derived from the finest-scale boundary feature, P_c itself, and a Sobel uncertainty estimate. The tanh bounds prevent overcorrection.',
    color: 'bg-green-50 border-green-200 text-green-900',
    accent: '#16a34a',
  },
]

export const modelStats = {
  totalParams: '65.40M',
  backboneParams: '50.14M',
  decoderParams: '13.03M',
  drbiParams: '2.02M',
  variant: 'MambaVision-S (small)',
  inputResolution: '256×256',
  output: 'Binary map',
}
