import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar.jsx'

function App() {
  const [players, setPlayers] = useState([])
  const [position, setPosition] = useState('ALL')
  const [search, setSearch] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [hoveredPlayer, setHoveredPlayer] = useState(null)

  useEffect(() => {
    fetch('http://localhost:8000/projections')
      .then(res => res.json())
      .then(data => {
        console.log('First Player:', data[0])
        const unique = data.filter((p, i, arr) =>
          arr.findIndex(x => x.player_id === p.player_id) === i
        )
        setPlayers(unique)
      })
  }, [])

  const rankedPlayers = players.map((player, index) => ({
    ...player,
    rank: index + 1
  }))

  const filteredPlayers = rankedPlayers
    .filter(p => position === 'ALL' || p.position === position)
    .filter(p => p.player_name.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="flex bg-gray-800 min-h-screen">

      {/* Sidebar — hidden on mobile, visible on md+ */}
      <Sidebar />

      {/* Mobile menu overlay */}
      {menuOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black opacity-50" onClick={() => setMenuOpen(false)} />
          <div className="absolute left-0 top-0 h-full w-64 z-10">
            <Sidebar />
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 p-4 md:p-8">

        {/* Mobile header with hamburger */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl md:text-4xl font-bold text-white">SnapCount</h1>
          <button
            className="md:hidden text-white bg-gray-700 p-2 rounded"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            ☰
          </button>
        </div>

        {/* Search bar */}
        <input
          type="text"
          placeholder="Search players..."
          onChange={e => setSearch(e.target.value)}
          className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 mb-4 outline-none focus:ring-2 focus:ring-blue-500"
        />

        {/* Position filters */}
        <div className="mb-6 flex flex-wrap gap-2">
          {['ALL', 'QB', 'RB', 'WR', 'TE'].map(pos => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`px-4 py-1 rounded-full text-sm font-medium ${pos === position ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
            >
              {pos}
            </button>
          ))}
        </div>

        {/* Table + hover panel */}
        <div className="flex gap-6">

          {/* Table */}
          <div className="flex-1 overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-300">
              <thead className="text-xs text-gray-400 uppercase bg-gray-700">
                <tr>
                  <th className="px-4 py-3">Rank</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3 hidden sm:table-cell">Position</th>
                  <th className="px-4 py-3 hidden sm:table-cell">Age</th>
                  <th className="px-4 py-3 hidden sm:table-cell">Team</th>
                  <th className="px-4 py-3">Proj PPR</th>
                  <th className="px-4 py-3 hidden md:table-cell">Sleeper ADP</th>
                  <th className="px-4 py-3 hidden md:table-cell">ADP Change</th>
                </tr>
              </thead>
              <tbody>
                {filteredPlayers.map(player => (
                  <tr
                    key={player.player_id}
                    className="border-b border-gray-700 hover:bg-gray-700 cursor-pointer"
                    onMouseEnter={() => setHoveredPlayer(player)}
                    onMouseLeave={() => setHoveredPlayer(null)}
                  >
                    <td className="px-4 py-3 text-gray-400">{player.rank}</td>
                    <td className="px-4 py-3 font-medium text-white">{player.player_name}</td>
                    <td className="px-4 py-3 hidden sm:table-cell">{player.position}</td>
                    <td className="px-4 py-3 hidden sm:table-cell">{player.age}</td>
                    <td className="px-4 py-3 hidden sm:table-cell">{player.team}</td>
                    <td className="px-4 py-3">{player.projected_pts_ppr?.toFixed(2)}</td>
                    <td className="px-4 py-3 hidden md:table-cell">{player.adp?.toFixed(0) ?? '-'}</td>
                    <td
                      className="px-4 py-3 hidden md:table-cell"
                      style={{ color: player.adp_value_ppr > 0 ? '#4ade80' : player.adp_value_ppr < 0 ? '#f87171' : 'inherit' }}
                    >
                      {player.adp_value_ppr?.toFixed(1) ?? '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Hover panel — shows on desktop when a player is hovered */}
          {hoveredPlayer && (
            <div className="hidden lg:block w-72 bg-gray-900 rounded-xl p-6 h-fit sticky top-8">
              <img
                src={hoveredPlayer.headshot_url}
                alt={hoveredPlayer.player_name}
                className="w-32 h-32 rounded-full mx-auto mb-4 object-cover bg-gray-700"
                onError={e => e.target.style.display = 'none'}
              />
              <h2 className="text-white text-xl font-bold text-center">{hoveredPlayer.player_name}</h2>
              <p className="text-gray-400 text-center mb-4">{hoveredPlayer.position} · {hoveredPlayer.team}</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Projected PPR</span>
                  <span className="text-white font-medium">{hoveredPlayer.projected_pts_ppr?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">PPG Last Season</span>
                  <span className="text-white font-medium">{hoveredPlayer.ppg_last_season ?? '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">VORP</span>
                  <span className="text-white font-medium">{hoveredPlayer.vorp_ppr?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Sleeper ADP</span>
                  <span className="text-white font-medium">{hoveredPlayer.adp?.toFixed(0) ?? '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">ADP Value</span>
                  <span
                    className="font-medium"
                    style={{ color: hoveredPlayer.adp_value_ppr > 0 ? '#4ade80' : hoveredPlayer.adp_value_ppr < 0 ? '#f87171' : 'white' }}
                  >
                    {hoveredPlayer.adp_value_ppr?.toFixed(1) ?? '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Conf. Range</span>
                  <span className="text-white font-medium">
                    {hoveredPlayer.confidence_low?.toFixed(1)} – {hoveredPlayer.confidence_high?.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}

export default App
