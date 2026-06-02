function Sidebar() {
  return (
    <div className="hidden md:flex bg-gray-900 h-screen w-64 flex-col p-6 shrink-0">
      <h1 className="text-white text-2xl font-bold mb-10">SnapCount</h1>
      <nav className="flex flex-col gap-3">
        <a className="text-white bg-gray-700 rounded px-4 py-2">Draft Board</a>
        <a className="text-gray-400 hover:text-white px-4 py-2">Trade Calculator</a>
      </nav>
    </div>
  )
}

export default Sidebar