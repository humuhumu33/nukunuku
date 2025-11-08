import React from 'react';
import ParallelUniverseExplorer from './ParallelUniverseExplorer';

const App = () => {
  return (
    <>
      <header className="container">
        <div className="header-logo-container">
          <div className="hologram-logo">
            <div className="iching-water-symbol">
              <div className="iching-line iching-line-broken"></div>
              <div className="iching-line iching-line-solid"></div>
              <div className="iching-line iching-line-broken"></div>
            </div>
          </div>
          <div className="header-brand-text">
            <div className="brand-name">Hologram</div>
            <div className="brand-taglines">
              <div className="tagline-left">VIRTUAL INFRASTRUCTURE FOR SCALABLE AI</div>
              <div className="tagline-bottom">
                <span className="tagline-digital">DIGITAL</span>
                <span className="tagline-physics">PHYSICS</span>
              </div>
            </div>
          </div>
        </div>
        <h1>Parallel Universe Explorer</h1>
        <p>Explore combinatorial spaces through symbolic computation</p>
      </header>
      <main className="container main-content">
        <ParallelUniverseExplorer />
      </main>
      <footer className="container">
        <div className="footer-email">🐇 trinity@uor.foundation</div>
      </footer>
    </>
  );
};

export default App;
