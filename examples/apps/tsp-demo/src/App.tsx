import React, { useState } from 'react';
import TSPDemo from './TSPDemo';
import PackingDemo from './PackingDemo';

const App = () => {
  const [activeDemo, setActiveDemo] = useState<'tsp' | 'packing'>('tsp');

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
        <p>Solve NP-hard problems through symbolic polynominal time computation.</p>
        
        {/* Demo Toggle */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '1rem',
          marginTop: '1.5rem',
          marginBottom: '1rem',
        }}>
          <button
            onClick={() => setActiveDemo('tsp')}
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '1rem',
              fontFamily: 'var(--font-family-mono)',
              backgroundColor: activeDemo === 'tsp' ? 'var(--primary-color)' : 'var(--surface-color)',
              color: activeDemo === 'tsp' ? 'var(--bg-color)' : 'var(--text-color)',
              border: '1px solid var(--border-color)',
              cursor: 'pointer',
              transition: 'all 0.2s linear',
              letterSpacing: 'var(--letter-spacing-tight)',
            }}
            onMouseEnter={(e) => {
              if (activeDemo !== 'tsp') {
                e.currentTarget.style.backgroundColor = 'var(--surface-elevated)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeDemo !== 'tsp') {
                e.currentTarget.style.backgroundColor = 'var(--surface-color)';
              }
            }}
          >
            Traveling Salesman
          </button>
          <button
            onClick={() => setActiveDemo('packing')}
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '1rem',
              fontFamily: 'var(--font-family-mono)',
              backgroundColor: activeDemo === 'packing' ? 'var(--primary-color)' : 'var(--surface-color)',
              color: activeDemo === 'packing' ? 'var(--bg-color)' : 'var(--text-color)',
              border: '1px solid var(--border-color)',
              cursor: 'pointer',
              transition: 'all 0.2s linear',
              letterSpacing: 'var(--letter-spacing-tight)',
            }}
            onMouseEnter={(e) => {
              if (activeDemo !== 'packing') {
                e.currentTarget.style.backgroundColor = 'var(--surface-elevated)';
              }
            }}
            onMouseLeave={(e) => {
              if (activeDemo !== 'packing') {
                e.currentTarget.style.backgroundColor = 'var(--surface-color)';
              }
            }}
          >
            Packing Problem
          </button>
        </div>
      </header>
      <main className="container main-content">
        {activeDemo === 'tsp' ? <TSPDemo /> : <PackingDemo />}
      </main>
      <footer className="container">
        <div className="footer-email">🐇 trinity@uor.foundation</div>
      </footer>
    </>
  );
};

export default App;
