import { useState, useEffect } from 'react'

const NAV_ITEMS = [
  { href: '#overview', label: 'Overview' },
  { href: '#motivation', label: 'Motivation' },
  { href: '#architecture', label: 'Architecture' },
  { href: '#math', label: 'Math Intuition' },
  { href: '#datasets', label: 'Datasets' },
  { href: '#results', label: 'Results' },
  { href: '#ablation', label: 'Ablation' },
  { href: '#scaling', label: 'Backbone' },
  { href: '#comparison', label: 'Comparison' },
  { href: '#learned', label: 'Learned' },
  { href: '#limitations', label: 'Next Steps' },
  { href: '#contact', label: 'Repo' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-200 ${
        scrolled ? 'bg-white/95 backdrop-blur border-b border-border shadow-sm' : 'bg-transparent'
      }`}
    >
      <nav className="max-w-container mx-auto px-4 sm:px-6 flex items-center justify-between h-14">
        <a href="#overview" className="font-bold text-main-text text-sm tracking-tight">
          Mamba<span className="text-primary">Refine</span>-CD
        </a>

        {/* Desktop nav */}
        <ul className="hidden xl:flex items-center gap-0.5">
          {NAV_ITEMS.map((item) => (
            <li key={item.href}>
              <a
                href={item.href}
                className="px-3 py-1.5 text-xs font-medium text-muted-text hover:text-primary rounded-md hover:bg-soft-blue/50 transition-colors"
              >
                {item.label}
              </a>
            </li>
          ))}
        </ul>

        {/* Mobile hamburger */}
        <button
          className="xl:hidden p-2 rounded-md hover:bg-slate-100"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span className={`block w-5 h-0.5 bg-main-text transition-all ${menuOpen ? 'rotate-45 translate-y-1' : ''}`} />
          <span className={`block w-5 h-0.5 bg-main-text mt-1 transition-all ${menuOpen ? 'opacity-0' : ''}`} />
          <span className={`block w-5 h-0.5 bg-main-text mt-1 transition-all ${menuOpen ? '-rotate-45 -translate-y-1.5' : ''}`} />
        </button>
      </nav>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="xl:hidden bg-white border-b border-border px-4 pb-4">
          <ul className="grid grid-cols-3 gap-1 pt-2">
            {NAV_ITEMS.map((item) => (
              <li key={item.href}>
                <a
                  href={item.href}
                  onClick={() => setMenuOpen(false)}
                  className="block px-3 py-2 text-xs font-medium text-muted-text hover:text-primary hover:bg-soft-blue/50 rounded-md transition-colors"
                >
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </header>
  )
}
