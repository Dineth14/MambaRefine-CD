import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

interface MathBlockProps {
  tex: string
  inline?: boolean
  className?: string
}

export default function MathBlock({ tex, inline = false, className = '' }: MathBlockProps) {
  if (inline) return <InlineMath math={tex} />
  return (
    <div className={`overflow-x-auto py-3 ${className}`}>
      <BlockMath math={tex} />
    </div>
  )
}
