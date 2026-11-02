import React, { useState, useEffect } from "react";
import axios from "axios";

const API = "http://localhost:8000";

const STEP_ICONS = ["🎬", "🔊", "👤", "💬", "✅"];
const STEP_NAMES = [
  "Splitting video",
  "Audio extraction",
  "Face extraction",
  "Text extraction",
  "Complete",
];

const FEATURE_DESCRIPTIONS = {
  audio_pause_count: "Number of silent pauses",
  audio_pause_mean_duration: "Average pause length (seconds)",
  audio_pause_to_speech_ratio: "Silence ÷ speech time",
  audio_speech_duration: "Total speaking time (seconds)",
  audio_total_duration: "Recording length (seconds)",
  audio_speech_rate_wpm: "Words per minute",
  audio_f0_mean: "Average voice pitch (Hz)",
  audio_f0_std: "Pitch variability",
  audio_f0_range: "Pitch range (Hz)",
  audio_loudness_mean: "Average volume",
  audio_loudness_std: "Volume variability",
  audio_jitter: "Voice pitch instability",
  audio_shimmer: "Voice amplitude instability",
  face_au04_mean: "Brow lowerer (furrowing)",
  face_au04_std: "Brow lowerer variability",
  face_au06_mean: "Cheek raiser (smile)",
  face_au06_std: "Cheek raiser variability",
  face_au12_mean: "Lip corner pull (smile)",
  face_au12_std: "Lip corner pull variability",
  face_au15_mean: "Lip corner depress (frown)",
  face_au15_std: "Lip corner depress variability",
  face_au_total_activity: "Overall face expressiveness",
  face_blink_rate_per_min: "Blinks per minute",
  face_detection_rate: "Face detected in % of frames",
  face_gaze_direction_variance: "Eye movement amount",
  face_head_movement_velocity: "Head movement speed",
  face_quality_warning: "Quality warning (1 = low quality)",
  text_word_count: "Total words spoken",
  text_first_person_pronoun_ratio: "I/me/my usage ratio",
  text_sentiment_positive: "Positive tone score",
  text_sentiment_negative: "Negative tone score",
  text_sentiment_neutral: "Neutral tone score",
  text_mean_utterance_length: "Average words per sentence",
  text_lexical_diversity: "Vocabulary variety (0-1)",
  text_speaking_time_seconds: "Speaking duration (seconds)",
  text_language: "Language (0=English, 1=Hindi, 2=Marathi)",
};

function ProgressSection({ status }) {
  const currentStep = status.current_step || 0;
  const progress = Math.min((currentStep / status.total_steps) * 100, 100);

  return (
    <div className="progress-section">
      <h2>Processing Pipeline</h2>
      <p className="filename">File: {status.filename}</p>

      <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>
      <p className="step-label">{status.step_label}</p>

      <div className="steps-grid">
        {STEP_NAMES.slice(0, 4).map((name, i) => {
          const stepNum = i + 1;
          const isDone = status.steps_done?.find(
            (s) => s.name === ["Split", "Audio", "Face", "Text"][i]
          );
          const isActive = currentStep === stepNum;
          const isPending = currentStep < stepNum;

          let className = "step-card";
          if (isDone) className += " done";
          else if (isActive) className += " active";
          else if (isPending) className += " pending";

          return (
            <div key={i} className={className}>
              <div className="step-icon">{STEP_ICONS[i]}</div>
              <div className="step-name">{name}</div>
              {isDone && (
                <div className="step-count">{isDone.count} features</div>
              )}
              {isActive && <div className="step-spinner"></div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SummaryCards({ summary }) {
  return (
    <div className="summary-section">
      <h2>Summary</h2>
      <div className="summary-grid">
        <div className="summary-card face-card">
          <h3>👤 Face</h3>
          <div className="stat">
            <span className="stat-label">Detected</span>
            <span className="stat-value">{summary.face_detected}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Smile (AU12)</span>
            <span className="stat-value">{summary.smile}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Frown (AU15)</span>
            <span className="stat-value">{summary.frown}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Blink rate</span>
            <span className="stat-value">{summary.blink_rate}</span>
          </div>
        </div>

        <div className="summary-card voice-card">
          <h3>🔊 Voice</h3>
          <div className="stat">
            <span className="stat-label">Pitch</span>
            <span className="stat-value">{summary.pitch}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Pauses</span>
            <span className="stat-value">{summary.pauses}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Speaking</span>
            <span className="stat-value">
              {summary.speech_duration} / {summary.total_duration}
            </span>
          </div>
        </div>

        <div className="summary-card text-card">
          <h3>💬 Speech</h3>
          <div className="stat">
            <span className="stat-label">Words</span>
            <span className="stat-value">{summary.words}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Negative</span>
            <span className="stat-value">{summary.sentiment_negative}</span>
          </div>
          <div className="stat">
            <span className="stat-label">I/me/my</span>
            <span className="stat-value">{summary.pronoun_ratio}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function FeatureTable({ title, features, icon }) {
  const [expanded, setExpanded] = useState(false);
  const entries = Object.entries(features);
  if (entries.length === 0) return null;

  return (
    <div className="feature-table-section">
      <div
        className="feature-table-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span>
          {icon} {title} ({entries.length} features)
        </span>
        <span>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <table className="feature-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key}>
                <td className="feature-key">{key}</td>
                <td className="feature-value">
                  {typeof value === "number" ? value.toFixed(4) : value}
                </td>
                <td className="feature-desc">
                  {FEATURE_DESCRIPTIONS[key] || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function ResultsPage({ sessionId, onBack }) {
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);

  useEffect(() => {
    if (!sessionId) return;

    const poll = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/status/${sessionId}`);
        setStatus(res.data);

        if (res.data.status === "complete") {
          clearInterval(poll);
          const resData = await axios.get(`${API}/results/${sessionId}`);
          setResults(resData.data);
        } else if (res.data.status === "failed") {
          clearInterval(poll);
        }
      } catch (err) {
        console.error("Poll error:", err);
      }
    }, 1500);

    return () => clearInterval(poll);
  }, [sessionId]);

  if (!status) {
    return (
      <div className="results-container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Connecting to server...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="results-container">
      <button className="btn btn-back" onClick={onBack}>
        ← Back
      </button>

      <ProgressSection status={status} />

      {status.status === "failed" && (
        <div className="error-box">
          <h3>Processing Failed</h3>
          <p>{status.step_label}</p>
        </div>
      )}

      {results && (
        <>
          <SummaryCards summary={results.summary} />

          <div className="all-features">
            <h2>All {results.total_features} Features</h2>
            <FeatureTable
              title="Audio Features"
              features={results.audio_features}
              icon="🔊"
            />
            <FeatureTable
              title="Face Features"
              features={results.face_features}
              icon="👤"
            />
            <FeatureTable
              title="Text Features"
              features={results.text_features}
              icon="💬"
            />
          </div>

          {results.warnings.length > 0 && (
            <div className="warnings-box">
              <h3>Warnings</h3>
              {results.warnings.map((w, i) => (
                <p key={i}>⚠️ {w}</p>
              ))}
            </div>
          )}

          <div className="disclaimer">
            This is a screening indicator, not a diagnosis. Please consult a
            qualified mental health professional.
          </div>
        </>
      )}
    </div>
  );
}
