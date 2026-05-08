export default function DRBIDiagram() {
  return (
    <div className="card p-6">
      <h3 className="text-sm font-semibold text-muted-text mb-4 uppercase tracking-wide">D-RBI: Differential Region-Boundary Interaction</h3>
      <div className="overflow-x-auto">
        <svg viewBox="0 0 700 320" className="w-full max-w-2xl mx-auto" aria-label="D-RBI module diagram">
          {/* Feature inputs */}
          <rect x="10" y="100" width="80" height="40" rx="8" fill="#dbeafe" stroke="#93c5fd" strokeWidth="1.5" />
          <text x="50" y="125" textAnchor="middle" fontSize="11" fill="#1e40af" fontWeight="600">F1^s</text>

          <rect x="10" y="180" width="80" height="40" rx="8" fill="#dbeafe" stroke="#93c5fd" strokeWidth="1.5" />
          <text x="50" y="205" textAnchor="middle" fontSize="11" fill="#1e40af" fontWeight="600">F2^s</text>

          {/* Arrow to diff ops */}
          <line x1="90" y1="120" x2="150" y2="140" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
          <line x1="90" y1="200" x2="150" y2="180" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />

          {/* Diff operations box */}
          <rect x="155" y="110" width="130" height="100" rx="10" fill="#f0fdf4" stroke="#86efac" strokeWidth="1.5" />
          <text x="220" y="132" textAnchor="middle" fontSize="11" fill="#166534" fontWeight="600">GroupNorm + concat</text>
          <text x="220" y="152" textAnchor="middle" fontSize="10" fill="#16a34a">[F1, F2]</text>
          <text x="220" y="167" textAnchor="middle" fontSize="10" fill="#16a34a">|F2 - F1|</text>
          <text x="220" y="182" textAnchor="middle" fontSize="10" fill="#16a34a">F2 - F1</text>
          <text x="220" y="197" textAnchor="middle" fontSize="10" fill="#16a34a">compress to D^s</text>

          {/* Arrow to region branch */}
          <line x1="285" y1="130" x2="350" y2="115" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
          {/* Arrow to boundary branch */}
          <line x1="285" y1="185" x2="350" y2="205" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />

          {/* Region branch */}
          <rect x="355" y="90" width="110" height="50" rx="8" fill="#fef9c3" stroke="#fde047" strokeWidth="1.5" />
          <text x="410" y="111" textAnchor="middle" fontSize="11" fill="#713f12" fontWeight="600">Region gate</text>
          <text x="410" y="126" textAnchor="middle" fontSize="10" fill="#854d0e">[0.2, 0.8]</text>

          {/* Boundary branch */}
          <rect x="355" y="185" width="110" height="50" rx="8" fill="#fce7f3" stroke="#f9a8d4" strokeWidth="1.5" />
          <text x="410" y="206" textAnchor="middle" fontSize="11" fill="#831843" fontWeight="600">Boundary gate</text>
          <text x="410" y="221" textAnchor="middle" fontSize="10" fill="#9d174d">Sobel, [0.0, 0.4]</text>

          {/* Arrows to fusion */}
          <line x1="465" y1="115" x2="540" y2="150" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
          <line x1="465" y1="210" x2="540" y2="170" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />

          {/* Outputs */}
          <rect x="545" y="130" width="100" height="60" rx="10" fill="#ede9fe" stroke="#c4b5fd" strokeWidth="1.5" />
          <text x="595" y="154" textAnchor="middle" fontSize="11" fill="#3730a3" fontWeight="600">Streams</text>
          <text x="595" y="170" textAnchor="middle" fontSize="10" fill="#4338ca">R^s and B^s</text>

          {/* Arrow to output */}
          <line x1="645" y1="160" x2="690" y2="160" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)" />
          <text x="690" y="155" textAnchor="start" fontSize="11" fill="#475569" fontWeight="600">to decoder</text>

          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8" />
            </marker>
          </defs>
        </svg>
      </div>
      <p className="text-xs text-muted-text text-center mt-2">
        D-RBI constructs explicit temporal evidence, compresses it to D^s, and produces related but separate
        region and boundary streams through bounded gates.
      </p>
    </div>
  )
}
