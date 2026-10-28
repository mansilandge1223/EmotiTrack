import React from "react";
import "./App.css";
export default function App() {
  return (
    <main className="app">
      <nav className="navbar"><h2>EmotiTrack</h2><span>AI-POWERED MONITORING</span></nav>
      <section className="hero">
        <p className="eyebrow">PERSONALIZED BEHAVIORAL MONITORING SYSTEM</p>
        <h1>Understand Behavioral Patterns</h1>
        <p className="subtitle">A centralized platform for recording, monitoring and analyzing behavioral patterns using audio and video data.</p>
        <div className="cards">
          <article><div className="emoji">◉</div><h3>Smart Monitoring</h3><p>Organize behavioral observations in one place.</p></article>
          <article><div className="emoji">✦</div><h3>AI Analysis</h3><p>Build tools to analyze audio and video patterns.</p></article>
          <article><div className="emoji">▥</div><h3>Visual Reports</h3><p>Present observations through clear reports.</p></article>
        </div>
        <p className="note">EmotiTrack supports observation and monitoring; AI outputs should not be treated as definitive emotion diagnoses.</p>
      </section>
    </main>
  );
}
