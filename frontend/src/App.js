import React, { useState } from "react";
import "./App.css";
import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";

export default function App() {
  const [page, setPage] = useState("home");
  const [sessionId, setSessionId] = useState(null);

  const handleUploadStart = (sid) => {
    setSessionId(sid);
    setPage("results");
  };

  return (
    <div className="app">
      <nav className="navbar">
        <h2 onClick={() => setPage("home")} style={{ cursor: "pointer" }}>
          EmotiTrack
        </h2>
        <span>MULTIMODAL FEATURE EXTRACTION</span>
      </nav>

      {page === "home" && (
        <div className="hero">
          <p className="eyebrow">DEMO DASHBOARD</p>
          <h1>Emotion &amp; Mental State Detection</h1>
          <p className="subtitle">
            Upload a video or record from your webcam. The system extracts 62
            features from face, voice, and speech in real-time.
          </p>
          <UploadPage onSessionStart={handleUploadStart} />
          <p className="note">
            This is a screening indicator, not a diagnosis. Please consult a
            qualified mental health professional.
          </p>
        </div>
      )}

      {page === "results" && sessionId && (
        <ResultsPage
          sessionId={sessionId}
          onBack={() => setPage("home")}
        />
      )}
    </div>
  );
}
