import Navbar from './components/Navbar'
import Home from './pages/Home'

export default function App() {
  return (
    <div className="min-h-screen bg-background text-main-text">
      <Navbar />
      <main>
        <Home />
      </main>
    </div>
  )
}
