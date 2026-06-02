import { useState, useEffect } from 'react'
import './App.css'

function App() {

  const [players, setPlayers] = useState([])
  const [position, setPosition] = useState('ALL')
  useEffect(() => {
    fetch('http://localhost:8000/projections')
      .then(res => res.json())
      .then(data => {
      const unique = data.filter((p, i, arr) => 
          arr.findIndex(x => x.player_id === p.player_id) === i
        )
        setPlayers(unique)
      })
  }, [])

  const filteredPlayers = position === 'ALL' 
    ? players 
    : players.filter(p => p.position === position)

  return (
    <div>
      <h1 className="text-4xl font-bold text-blue-500">Draft Scout</h1>
      <div>
      {['ALL', 'QB', 'RB', 'WR', 'TE'].map(pos => (
        <button key={pos} onClick={() => setPosition(pos)}>
          {pos}
        </button>
      ))}
      </div>
      <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Position</th>
          <th>Team</th>
          <th>Projected PPR</th>
        </tr>
      </thead>
      <tbody>
        {filteredPlayers.map(player => (
          <tr key={player.player_id}>
            <td>{player.player_name}</td>
            <td>{player.position}</td>
            <td>{player.team}</td>
            <td>{player.projected_pts_ppr?.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </div>

  )
}

export default App
