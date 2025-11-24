import { useState } from 'react';

function App() {
  return (
    <div className="bg-gray-900 text-white min-h-screen">
      <header className="bg-gray-800 p-4">
        <h1 className="text-xl">Config Comparator</h1>
      </header>
      <main className="p-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h2 className="text-lg mb-2">Config 1</h2>
            <div className="bg-gray-800 p-4 rounded">
              <p>Config file content will go here.</p>
            </div>
          </div>
          <div>
            <h2 className="text-lg mb-2">Config 2</h2>
            <div className="bg-gray-800 p-4 rounded">
              <p>Config file content will go here.</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;