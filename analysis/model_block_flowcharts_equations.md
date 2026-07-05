# MambaRefine-CD Model Blocks: Mermaid Flowcharts and LaTeX Equations

This document follows the current implementation in `src/models`.

Source grounding:
- MambaVision: <https://arxiv.org/abs/2407.08083>
- Mamba selective SSM: <https://arxiv.org/abs/2312.00752>
- Feature Pyramid Network: <https://arxiv.org/abs/1612.03144>
- Sobel operator reference: <https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm>
- OpenCV gradient/Sobel implementation reference: <https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html>

Implementation note: the current `src/models` code uses `TemporalDifference` modes (`abs_only`, `signed_only`, `abs_signed`) before D-RBI. It does not concatenate raw paired features `[F_a, F_b]` inside D-RBI, and it does not contain CRAM-lite in the active model path.

Provenance note: no exact arXiv/source paper was found for the names `D-RBI`, `Differential Region-Boundary Interaction`, `ARF-FPN` as implemented here, or `BoundaryRefinement` as implemented here. Those blocks are therefore documented from code, while the encoder, selective scan basis, FPN-style top-down fusion, and Sobel operator are grounded in external sources.

## 0. End-to-End Model Wiring

```mermaid
flowchart TD
  IA["Image A<br/>B x 3 x H x W"] --> ENC["Shared encoder E_theta"]
  IB["Image B<br/>B x 3 x H x W"] --> ENC

  ENC --> FA["Features A<br/>Fa0..Fa3"]
  ENC --> FB["Features B<br/>Fb0..Fb3"]

  FA --> TD["TemporalDifference per stage"]
  FB --> TD

  TD --> DRBI0["D-RBI stage 0"]
  TD --> DRBI1["D-RBI stage 1"]
  TD --> DRBI2["D-RBI stage 2"]
  TD --> DRBI3["D-RBI stage 3"]

  DRBI0 --> R0["Region R0"]
  DRBI1 --> R1["Region R1"]
  DRBI2 --> R2["Region R2"]
  DRBI3 --> R3["Region R3"]

  DRBI0 --> B0["Boundary B0"]
  DRBI1 --> B1["Boundary B1"]
  DRBI2 --> B2["Boundary B2"]
  DRBI3 --> B3["Boundary B3"]

  R0 --> ARF["Multi-scale ARF-FPN decoder"]
  R1 --> ARF
  R2 --> ARF
  R3 --> ARF

  ARF --> PC["Coarse logits / main_logits<br/>B x 1 x H x W"]

  B0 --> BRR["Boundary residual refinement"]
  B1 --> BRR
  B2 --> BRR
  B3 --> BRR
  PC --> BRR

  BRR --> PF["Final logits<br/>B x 1 x H x W"]
  BRR --> PB["Boundary logits<br/>B x 1 x H x W"]
  BRR --> RES["Residual<br/>B x 1 x H x W"]
```

```latex
\begin{aligned}
\{F_a^s\}_{s=0}^{3} &= E_\theta(I_a),\\
\{F_b^s\}_{s=0}^{3} &= E_\theta(I_b),\\
D_{\mathrm{in}}^s &= \operatorname{TemporalDifference}(F_a^s,F_b^s),\\
(R^s,B^s) &= \operatorname{DRBI}_s(D_{\mathrm{in}}^s),\\
P_c &= \operatorname{ARFFPN}(\{R^s\}_{s=0}^{3}),\\
(P_f,P_b,\Delta P) &= \operatorname{BoundaryRefine}(P_c,\{B^s\}_{s=0}^{3}).
\end{aligned}
```

## 1. Shared MambaVision Encoder: Four Stages

Default active config is `mambavision/small`: channels `[96, 192, 384, 768]`, depths `[3, 3, 7, 5]`, heads `[2, 4, 8, 16]`, windows `[8, 8, 14, 7]`.

```mermaid
flowchart LR
  I["Input image<br/>B x 3 x H x W"] --> PE["PatchEmbed<br/>3x3 s2 Conv-BN-ReLU<br/>3x3 s2 Conv-BN-ReLU<br/>B x C0 x H/4 x W/4"]

  PE --> S0B

  subgraph S0["Stage 0 / levels.0"]
    S0B["ConvBlock x d0<br/>default small: d0=3<br/>C0=96"] --> F0["Output F0<br/>B x C0 x H/4 x W/4"]
    F0 --> DS0["Downsample<br/>3x3 s2 Conv<br/>C0 -> C1"]
  end

  DS0 --> S1B

  subgraph S1["Stage 1 / levels.1"]
    S1B["ConvBlock x d1<br/>default small: d1=3<br/>C1=192"] --> F1["Output F1<br/>B x C1 x H/8 x W/8"]
    F1 --> DS1["Downsample<br/>3x3 s2 Conv<br/>C1 -> C2"]
  end

  DS1 --> WP2

  subgraph S2["Stage 2 / levels.2"]
    WP2["Window partition<br/>window w2=14"] --> M2["MambaVision Block x ceil(d2/2)<br/>default small: 4 blocks"]
    M2 --> A2["Attention Block x floor(d2/2)<br/>default small: 3 blocks"]
    A2 --> WR2["Window reverse"]
    WR2 --> F2["Output F2<br/>B x C2 x H/16 x W/16"]
    F2 --> DS2["Downsample<br/>3x3 s2 Conv<br/>C2 -> C3"]
  end

  DS2 --> WP3

  subgraph S3["Stage 3 / levels.3"]
    WP3["Window partition<br/>window w3=7"] --> M3["MambaVision Block x ceil(d3/2)<br/>default small: 3 blocks"]
    M3 --> A3["Attention Block x floor(d3/2)<br/>default small: 2 blocks"]
    A3 --> WR3["Window reverse"]
    WR3 --> F3["Output F3<br/>B x C3 x H/32 x W/32"]
  end

  F0 --> OUT["Encoder returns<br/>{F0,F1,F2,F3}"]
  F1 --> OUT
  F2 --> OUT
  F3 --> OUT
```

```latex
\begin{aligned}
X_0 &= \rho\!\left(\operatorname{BN}\!\left(W_{2}^{3\times3,s=2}
      * \rho\!\left(\operatorname{BN}\!\left(W_{1}^{3\times3,s=2} * I\right)\right)\right)\right),\\
F^s &\in \mathbb{R}^{B \times C_s \times H_s \times W_s},\\
(H_s,W_s) &= (H/2^{s+2}, W/2^{s+2}), \qquad s\in\{0,1,2,3\}.
\end{aligned}
```

```latex
\begin{aligned}
\operatorname{ConvBlock}(X)
&= X + \operatorname{DropPath}\left(\gamma \odot
   \operatorname{BN}\left(W_2^{3\times3} *
   \operatorname{GELU}\left(\operatorname{BN}(W_1^{3\times3} * X)\right)\right)\right).
\end{aligned}
```

```latex
\begin{aligned}
X' &= X + \operatorname{DropPath}\left(\gamma_1 \odot
      \operatorname{Mixer}(\operatorname{LN}(X))\right),\\
X_{\mathrm{out}} &= X' + \operatorname{DropPath}\left(\gamma_2 \odot
      \operatorname{MLP}(\operatorname{LN}(X'))\right).
\end{aligned}
```

```latex
\begin{aligned}
X_{\mathrm{win}} &= \operatorname{WindowPartition}(X,w)
  \in \mathbb{R}^{B_w \times w^2 \times C},\\
X &= \operatorname{WindowReverse}(X_{\mathrm{win}},w,H,W).
\end{aligned}
```

### 1.1 MambaVision Mixer Equations

```mermaid
flowchart TD
  X["Window tokens X<br/>B_w x L x C"] --> IP["Linear in_proj"]
  IP --> SPLIT["Split channels"]
  SPLIT --> XS["SSM branch Xs"]
  SPLIT --> XZ["Symmetric branch Xz"]
  XS --> CX["same-pad depthwise Conv1D + SiLU"]
  CX --> XP["x_proj"]
  XP --> PARAM["dt, B, C parameters"]
  PARAM --> SCAN["SelectiveScan<br/>A=-exp(A_log), D skip"]
  XZ --> CZ["same-pad depthwise Conv1D + SiLU"]
  SCAN --> CAT["Concat"]
  CZ --> CAT
  CAT --> OP["Linear out_proj"]
  OP --> Y["Mixer output<br/>B_w x L x C"]
```

```latex
\begin{aligned}
[X_s,X_z] &= \operatorname{split}\left(W_{\mathrm{in}}X\right),\\
U_s &= \operatorname{SiLU}\left(\operatorname{Conv1D}_{\mathrm{same}}(X_s)\right),\\
U_z &= \operatorname{SiLU}\left(\operatorname{Conv1D}_{\mathrm{same}}(X_z)\right),\\
[\Delta_k,B_k,C_k] &= W_x U_{s,k},\\
A &= -\exp(A_{\log}),\\
Y_s &= \operatorname{SelectiveScan}(U_s;\Delta,A,B,C,D),\\
\operatorname{MambaVisionMixer}(X) &= W_{\mathrm{out}}\,[Y_s;U_z].
\end{aligned}
```

```latex
\begin{aligned}
h_k &= \bar{A}_k h_{k-1} + \bar{B}_k u_k,\\
y_k &= C_k h_k + D u_k,\\
\bar{A}_k &= \exp(\Delta_k A).
\end{aligned}
```

### 1.2 Attention Block Equations

```latex
\begin{aligned}
Q &= XW_Q,\qquad K=XW_K,\qquad V=XW_V,\\
\operatorname{Attention}(X)
&= W_O\left(\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V\right).
\end{aligned}
```

### 1.3 Variant Depth/Channel Summary From Code

```latex
\begin{array}{c|c|c|c}
\text{Variant} & (C_0,C_1,C_2,C_3) & (d_0,d_1,d_2,d_3) & \text{Stage 2/3 mixer split}\\
\hline
\text{T} & (80,160,320,640) & (1,3,8,4) & (4M+4A,\;2M+2A)\\
\text{T2} & (80,160,320,640) & (1,3,11,4) & (6M+5A,\;2M+2A)\\
\text{S} & (96,192,384,768) & (3,3,7,5) & (4M+3A,\;3M+2A)\\
\text{B} & (128,256,512,1024) & (3,3,10,5) & (5M+5A,\;3M+2A)\\
\text{L} & (196,392,784,1568) & (3,3,10,5) & (5M+5A,\;3M+2A)
\end{array}
```

## 2. D-RBI Block

### 2.1 Feature Difference/Concat and 1x1 Projection

```mermaid
flowchart TD
  FA["Stage feature Fa^s<br/>B x Cs x Hs x Ws"] --> DIFF["Delta^s = Fb^s - Fa^s"]
  FB["Stage feature Fb^s<br/>B x Cs x Hs x Ws"] --> DIFF
  DIFF --> MODE{"temporal_input_mode"}
  MODE --> ABS["abs_only<br/>D_in=abs(Delta)"]
  MODE --> SIGN["signed_only<br/>D_in=Delta"]
  MODE --> BOTH["abs_signed<br/>D_in=Concat(abs(Delta), Delta)"]
  ABS --> C1["1x1 Conv<br/>mult*Cs -> Cs"]
  SIGN --> C1
  BOTH --> C1
  C1 --> GN["GroupNorm"]
  GN --> GELU["GELU"]
  GELU --> SP["Spatial block<br/>depthwise 3x3 + pointwise 1x1 + GN + GELU"]
  SP --> D["Projected differential feature D^s<br/>B x Cs x Hs x Ws"]
```

```latex
\begin{aligned}
\Delta^s &= F_b^s - F_a^s,\\
D_{\mathrm{in}}^s &=
\begin{cases}
|\Delta^s|, & \texttt{abs\_only},\\
\Delta^s, & \texttt{signed\_only},\\
[|\Delta^s|;\Delta^s], & \texttt{abs\_signed},
\end{cases}\\
\hat{D}^s &= \operatorname{GELU}\left(\operatorname{GN}\left(W_{p}^{1\times1} * D_{\mathrm{in}}^s\right)\right),\\
D^s &= \operatorname{GELU}\left(\operatorname{GN}\left(W_{\mathrm{pw}}^{1\times1} *
      (W_{\mathrm{dw}}^{3\times3} * \hat{D}^s)\right)\right).
\end{aligned}
```

### 2.2 Regional Feature Path

```mermaid
flowchart TD
  D["D^s<br/>projected differential feature"] --> PHIR["phi_r gate MLP<br/>1x1 Conv -> GELU -> 1x1 Conv"]
  PHIR --> CLAMP["Clamp logits to [-8, 8]"]
  CLAMP --> SIG["Sigmoid"]
  SIG --> AFF["Affine gate range<br/>0.2 + 0.6 * sigmoid"]
  AFF --> GR["Region gate G_r^s<br/>range [0.2, 0.8]"]
  D --> MUL["Elementwise multiply"]
  GR --> MUL
  MUL --> R["Region feature R^s<br/>B x Cs x Hs x Ws"]
```

```latex
\begin{aligned}
\phi_r(D^s) &= W_{r,2}^{1\times1} *
              \operatorname{GELU}(W_{r,1}^{1\times1} * D^s + b_{r,1}) + b_{r,2},\\
G_r^s &= 0.2 + 0.6\cdot \sigma\left(\operatorname{clip}(\phi_r(D^s),-8,8)\right),\\
R^s &= D^s \odot G_r^s.
\end{aligned}
```

### 2.3 Boundary Feature Path

```mermaid
flowchart TD
  D["D^s<br/>projected differential feature"] --> SOB["SobelGradient<br/>per-channel grouped Sobel"]
  SOB --> E["Edge feature E^s"]
  E --> PHIB["phi_b gate MLP<br/>1x1 Conv -> GELU -> 1x1 Conv"]
  PHIB --> CLAMP["Clamp logits to [-8, 8]"]
  CLAMP --> SIG["Sigmoid"]
  SIG --> AFF["Affine gate range<br/>0.4 * sigmoid"]
  AFF --> GB["Boundary gate G_b^s<br/>range [0, 0.4]"]
  D --> MUL["Elementwise multiply"]
  GB --> MUL
  MUL --> B["Boundary feature B^s<br/>B x Cs x Hs x Ws"]
```

```latex
\begin{aligned}
E^s &= \operatorname{SobelGradient}(D^s),\\
\phi_b(E^s) &= W_{b,2}^{1\times1} *
              \operatorname{GELU}(W_{b,1}^{1\times1} * E^s + b_{b,1}) + b_{b,2},\\
G_b^s &= 0.4\cdot \sigma\left(\operatorname{clip}(\phi_b(E^s),-8,8)\right),\\
B^s &= D^s \odot G_b^s.
\end{aligned}
```

### 2.4 Sobel Block

```mermaid
flowchart TD
  X["Input x<br/>B x C x H x W"] --> KX["Grouped Conv2d with Kx<br/>one Sobel kernel per channel"]
  X --> KY["Grouped Conv2d with Ky<br/>one Sobel kernel per channel"]
  KX --> GX["g_x"]
  KY --> GY["g_y"]
  GX --> MAG["sqrt(g_x^2 + g_y^2 + 1e-6)"]
  GY --> MAG
  MAG --> CL["Clamp to [0, 10]"]
  CL --> OUT["SobelGradient(x)<br/>B x C x H x W"]
```

```latex
\begin{aligned}
K_x &=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\qquad
K_y =
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix},\\
G_x &= x * K_x,\qquad G_y = x * K_y,\\
\operatorname{SobelGradient}(x)
&= \operatorname{clip}\left(\sqrt{G_x^2+G_y^2+10^{-6}},0,10\right).
\end{aligned}
```

### 2.5 Phi Gate Block (`phi_r` / `phi_b`)

```mermaid
flowchart TD
  X["Input feature X<br/>D^s for region or E^s for boundary"] --> C1["1x1 Conv<br/>C -> h"]
  C1 --> GELU["GELU"]
  GELU --> C2["1x1 Conv<br/>h -> C"]
  C2 --> LOGITS["Gate logits"]
  LOGITS --> CLAMP["Clamp [-8, 8]"]
  CLAMP --> SIG["Sigmoid"]
  SIG --> RANGE{"Gate type"}
  RANGE --> R["Region: 0.2 + 0.6*sigma<br/>range [0.2,0.8]"]
  RANGE --> B["Boundary: 0.4*sigma<br/>range [0,0.4]"]
```

```latex
\begin{aligned}
h &= \max(\lfloor 0.25C\rfloor,1),\\
\phi(X) &= W_2^{1\times1} * \operatorname{GELU}(W_1^{1\times1} * X + b_1) + b_2,\\
\tilde{\phi}(X) &= \operatorname{clip}(\phi(X),-8,8),\\
G_r(X) &= 0.2 + 0.6\sigma(\tilde{\phi}(X)),\\
G_b(X) &= 0.4\sigma(\tilde{\phi}(X)).
\end{aligned}
```

## 3. Multi-Scale ARF-FPN Decoder

```mermaid
flowchart TD
  R0["R0<br/>B x C0 x H/4 x W/4"] --> P0["Proj0 1x1 ConvNormGELU<br/>B x D x H/4 x W/4"]
  R1["R1<br/>B x C1 x H/8 x W/8"] --> P1["Proj1 1x1 ConvNormGELU<br/>B x D x H/8 x W/8"]
  R2["R2<br/>B x C2 x H/16 x W/16"] --> P2["Proj2 1x1 ConvNormGELU<br/>B x D x H/16 x W/16"]
  R3["R3<br/>B x C3 x H/32 x W/32"] --> P3["Proj3 1x1 ConvNormGELU<br/>B x D x H/32 x W/32"]

  P0 --> A0["ARF block rates 1,2,4,8"]
  P1 --> A1["ARF block rates 1,2,4,8"]
  P2 --> A2["ARF block rates 1,2,4,8"]
  P3 --> A3["ARF block rates 1,2,4,8"]

  A3 --> T3["Smooth3"]
  T3 --> UP2["Upsample to stage 2"]
  A2 --> SUM2["Add"]
  UP2 --> SUM2
  SUM2 --> T2["Smooth2"]

  T2 --> UP1["Upsample to stage 1"]
  A1 --> SUM1["Add"]
  UP1 --> SUM1
  SUM1 --> T1["Smooth1"]

  T1 --> UP0["Upsample to stage 0"]
  A0 --> SUM0["Add"]
  UP0 --> SUM0
  SUM0 --> T0["Smooth0"]

  T0 --> HEAD["Coarse prediction head"]
  HEAD --> PC["main_logits P_c<br/>B x 1 x H x W"]
```

```latex
\begin{aligned}
\tilde{R}^s &= \operatorname{Proj}_{s}^{1\times1}(R^s),\\
Z_{s,r} &= \operatorname{ConvNormGELU}_{3\times3,d=r}(\tilde{R}^s),\qquad r\in\{1,2,4,8\},\\
\alpha_s &= \operatorname{softmax}\left(W_2\operatorname{ReLU}(W_1\operatorname{GAP}(\tilde{R}^s))\right),\\
P^s &= \sum_{r\in\{1,2,4,8\}}\alpha_{s,r}Z_{s,r}.
\end{aligned}
```

```latex
\begin{aligned}
T^3 &= \operatorname{Smooth}_3(P^3),\\
T^s &= \operatorname{Smooth}_s\left(P^s +
      \operatorname{Up}(T^{s+1};H_s,W_s)\right),\qquad s=2,1,0,\\
P_c &= \operatorname{Up}\left(\operatorname{Head}(T^0);H,W\right).
\end{aligned}
```

## 4. Coarse Prediction Head

```mermaid
flowchart TD
  T0["Top FPN feature T0<br/>B x D x H/4 x W/4"] --> CNG["ConvNormGELU 3x3<br/>D -> D/2"]
  CNG --> C1["1x1 Conv<br/>D/2 -> 1"]
  C1 --> UP["Bilinear upsample<br/>to H x W"]
  UP --> PC["Coarse logits P_c / main_logits<br/>B x 1 x H x W"]
```

```latex
\begin{aligned}
H_c &= \operatorname{GELU}\left(\operatorname{GN}\left(W_h^{3\times3} * T^0\right)\right),\\
\ell_c &= W_o^{1\times1} * H_c + b_o,\\
P_c &= \operatorname{BilinearUp}(\ell_c;H,W).
\end{aligned}
```

## 5. Boundary Residual Refinement

```mermaid
flowchart TD
  B0["Boundary B0"] --> AGG["Boundary aggregation"]
  B1["Boundary B1"] --> AGG
  B2["Boundary B2"] --> AGG
  B3["Boundary B3"] --> AGG
  AGG --> BF["b_feat<br/>B x D x H x W"]

  BF --> BH["Boundary head"]
  BH --> PB["boundary_logits<br/>B x 1 x H x W"]

  PC["main_logits P_c<br/>B x 1 x H x W"] --> SIG["Sigmoid"]
  SIG --> SOB["SobelEdge"]
  SOB --> EDGE["edge<br/>B x 1 x H x W"]

  BF --> CAT["Concat along channels"]
  PC --> CAT
  EDGE --> CAT
  CAT --> RH["Refinement head"]
  RH --> RAW["raw_delta"]
  RAW --> TANH["0.1 * tanh"]
  TANH --> RES["residual Delta P"]
  PC --> ADD["Elementwise addition"]
  RES --> ADD
  ADD --> PF["final_logits P_f"]
```

```latex
\begin{aligned}
F_b &= \operatorname{BoundaryAggregate}(\{B^s\}_{s=0}^{3}),\\
P_b &= \operatorname{BoundaryHead}(F_b),\\
E_c &= \operatorname{SobelEdge}(\sigma(P_c)),\\
\Delta_{\mathrm{raw}} &= \operatorname{RefineHead}([F_b;P_c;E_c]),\\
\Delta P &= 0.1\tanh(\Delta_{\mathrm{raw}}),\\
P_f &= P_c + \Delta P.
\end{aligned}
```

### 5.1 Boundary Aggregation Block

```mermaid
flowchart TD
  B0["B0"] --> P0["1x1 ConvNormGELU"]
  B1["B1"] --> P1["1x1 ConvNormGELU"]
  B2["B2"] --> P2["1x1 ConvNormGELU"]
  B3["B3"] --> P3["1x1 ConvNormGELU"]

  P0 --> U0["Upsample to H x W"]
  P1 --> U1["Upsample to H x W"]
  P2 --> U2["Upsample to H x W"]
  P3 --> U3["Upsample to H x W"]

  U0 --> AVG["Average"]
  U1 --> AVG
  U2 --> AVG
  U3 --> AVG
  AVG --> SM["Smooth ConvNormGELU 3x3"]
  SM --> BF["b_feat<br/>B x D x H x W"]
```

```latex
\begin{aligned}
Q^s &= \operatorname{Up}\left(\operatorname{Proj}_{b,s}^{1\times1}(B^s);H,W\right),\\
F_b &= \operatorname{Smooth}\left(\frac{1}{4}\sum_{s=0}^{3} Q^s\right).
\end{aligned}
```

### 5.2 Sobel Operator in Refinement

```mermaid
flowchart TD
  PC["main_logits P_c"] --> SIG["sigmoid(P_c)"]
  SIG --> KX["Conv2d Kx"]
  SIG --> KY["Conv2d Ky"]
  KX --> GX["g_x"]
  KY --> GY["g_y"]
  GX --> MAG["sqrt(g_x^2 + g_y^2 + 1e-6)"]
  GY --> MAG
  MAG --> EDGE["edge E_c<br/>B x 1 x H x W"]
```

```latex
\begin{aligned}
P &= \sigma(P_c),\\
G_x &= P * K_x,\qquad G_y = P * K_y,\\
E_c &= \sqrt{G_x^2 + G_y^2 + 10^{-6}}.
\end{aligned}
```

### 5.3 Refinement Head

```mermaid
flowchart TD
  BF["b_feat<br/>B x D x H x W"] --> CAT["Concat"]
  PC["P_c<br/>B x 1 x H x W"] --> CAT
  EDGE["E_c<br/>B x 1 x H x W"] --> CAT
  CAT --> X["B x (D+2) x H x W"]
  X --> DWPW["Depthwise-pointwise 3x3 block<br/>(or ConvNormGELU if disabled)"]
  DWPW --> C1["1x1 Conv<br/>D -> 1"]
  C1 --> RAW["raw_delta"]
```

```latex
\begin{aligned}
X_r &= [F_b;P_c;E_c]\in \mathbb{R}^{B\times(D+2)\times H\times W},\\
H_r &= \operatorname{GELU}\left(\operatorname{GN}\left(W_{\mathrm{pw}}^{1\times1} *
      (W_{\mathrm{dw}}^{3\times3} * X_r)\right)\right),\\
\Delta_{\mathrm{raw}} &= W_{\Delta}^{1\times1} * H_r + b_{\Delta}.
\end{aligned}
```

### 5.4 Elementwise Addition Block

```mermaid
flowchart LR
  RAW["raw_delta"] --> TANH["tanh"]
  TANH --> SCALE["scale by 0.1"]
  SCALE --> RES["residual Delta P"]
  PC["main_logits P_c"] --> ADD["Elementwise add"]
  RES --> ADD
  ADD --> PF["final_logits P_f"]
```

```latex
\begin{aligned}
\Delta P &= 0.1\tanh(\Delta_{\mathrm{raw}}),\\
P_f &= P_c + \Delta P.
\end{aligned}
```

## 6. Shared Helper Blocks in `heads.py`

```mermaid
flowchart TD
  X["Input x"] --> CONV["Conv2d"]
  CONV --> GN["GroupNorm<br/>largest divisor <= 32"]
  GN --> GELU["GELU"]
  GELU --> Y["ConvNormGELU(x)"]
```

```latex
\begin{aligned}
\operatorname{groups}(C) &= \max\{g: g\le 32,\; C\bmod g=0\},\\
\operatorname{ConvNormGELU}(x) &= \operatorname{GELU}(\operatorname{GN}(W*x)).
\end{aligned}
```

```mermaid
flowchart TD
  X["Input x"] --> DW["Depthwise 3x3 Conv<br/>groups=C_in"]
  DW --> PW["Pointwise 1x1 Conv"]
  PW --> GN["GroupNorm"]
  GN --> GELU["GELU"]
  GELU --> Y["depthwise_pointwise(x)"]
```

```latex
\begin{aligned}
\operatorname{DWConv}(x)_c &= K_c^{3\times3} * x_c,\\
\operatorname{DepthwisePointwise}(x)
&= \operatorname{GELU}\left(\operatorname{GN}\left(W_{\mathrm{pw}}^{1\times1} *
\operatorname{DWConv}(x)\right)\right).
\end{aligned}
```

## 7. Optional VMamba Adapter in the Same Folder

The active config uses MambaVision, but `src/models/encoders/vmamba_adapter.py` is also in the folder.

```mermaid
flowchart TD
  I["Input image"] --> PE["VMamba patch_embed"]
  PE --> L0["layer 0 blocks"]
  L0 --> F0["append BCHW feature 0"]
  F0 --> DS0["downsample if present"]
  DS0 --> L1["layer 1 blocks"]
  L1 --> F1["append BCHW feature 1"]
  F1 --> DS1["downsample if present"]
  DS1 --> L2["layer 2 blocks"]
  L2 --> F2["append BCHW feature 2"]
  F2 --> DS2["downsample if present"]
  DS2 --> L3["layer 3 blocks"]
  L3 --> F3["append BCHW feature 3"]
  F0 --> OUT["return F0..F3"]
  F1 --> OUT
  F2 --> OUT
  F3 --> OUT
```

```latex
\begin{aligned}
X_0 &= \operatorname{patch\_embed}(I),\\
X_{s+1},F^s &= \operatorname{VMambaLayer}_s(X_s),\\
\{F^0,F^1,F^2,F^3\} &= \operatorname{VMambaAdapter}(I).
\end{aligned}
```
