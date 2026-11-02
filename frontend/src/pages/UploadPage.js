import React, { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:8000";
const MAX_RECORD_SECONDS = 60;

const QUESTIONS = [
  "How are you feeling today? Describe your mood.",
  "Tell me about your sleep — have you been sleeping well lately?",
  "What activities have you enjoyed recently?",
  "Is there anything that has been worrying you?",
  "How would you describe your energy level this week?",
];

export default function UploadPage({ onSessionStart }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [timeLeft, setTimeLeft] = useState(MAX_RECORD_SECONDS);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const videoRef = useRef(null);
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const fileInputRef = useRef(null);
  const timerRef = useRef(null);
  const questionTimerRef = useRef(null);

  const stopRecording = useCallback(() => {
    if (mediaRecorder.current && mediaRecorder.current.state === "recording") {
      mediaRecorder.current.stop();
    }
    setRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
    if (questionTimerRef.current) clearInterval(questionTimerRef.current);
  }, []);

  // Auto-stop when timer hits 0
  useEffect(() => {
    if (recording && timeLeft <= 0) {
      stopRecording();
    }
  }, [timeLeft, recording, stopRecording]);

  const uploadFile = async (file) => {
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post(`${API}/upload`, form);
      onSessionStart(res.data.session_id);
    } catch (err) {
      alert("Upload failed: " + (err.response?.data?.detail || err.message));
      setUploading(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;
      videoRef.current.play();

      const recorder = new MediaRecorder(stream);
      mediaRecorder.current = recorder;
      chunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: "video/webm" });
        setRecordedBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
        if (videoRef.current) videoRef.current.srcObject = null;
      };

      recorder.start();
      setRecording(true);
      setTimeLeft(MAX_RECORD_SECONDS);
      setCurrentQuestion(0);

      // Countdown timer
      timerRef.current = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);

      // Cycle questions every 12 seconds
      questionTimerRef.current = setInterval(() => {
        setCurrentQuestion((prev) =>
          prev < QUESTIONS.length - 1 ? prev + 1 : prev
        );
      }, 12000);
    } catch (err) {
      alert("Camera access denied: " + err.message);
    }
  };

  const uploadRecording = () => {
    if (recordedBlob) {
      const file = new File([recordedBlob], "recording.webm", {
        type: "video/webm",
      });
      uploadFile(file);
      setRecordedBlob(null);
    }
  };

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (uploading) {
    return (
      <div className="upload-status">
        <div className="spinner"></div>
        <p>Uploading video...</p>
      </div>
    );
  }

  return (
    <div className="upload-section">
      <div className="upload-options">
        {/* File Upload */}
        <div
          className={`upload-box ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">📁</div>
          <h3>Upload Video</h3>
          <p>Drag & drop or click to select</p>
          <p className="file-types">.mp4, .avi, .mov, .webm</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />
        </div>

        {/* Webcam Record */}
        <div className="upload-box record-box">
          <div className="upload-icon">🎥</div>
          <h3>Record from Webcam</h3>

          {/* Timer */}
          {recording && (
            <div className="timer-bar">
              <div
                className={`timer-display ${timeLeft <= 10 ? "timer-warning" : ""}`}
              >
                {formatTime(timeLeft)}
              </div>
              <div className="timer-track">
                <div
                  className="timer-fill"
                  style={{
                    width: `${((MAX_RECORD_SECONDS - timeLeft) / MAX_RECORD_SECONDS) * 100}%`,
                  }}
                ></div>
              </div>
            </div>
          )}

          {/* Video Preview */}
          <video
            ref={videoRef}
            className="preview-video"
            muted
            style={{ display: recording ? "block" : "none" }}
          />

          {/* Question Prompt */}
          {recording && (
            <div className="question-prompt">
              <div className="question-label">
                Question {currentQuestion + 1} of {QUESTIONS.length}
              </div>
              <div className="question-text">
                "{QUESTIONS[currentQuestion]}"
              </div>
            </div>
          )}

          {/* Controls */}
          {!recording && !recordedBlob && (
            <div>
              <button className="btn btn-primary" onClick={startRecording}>
                Start Recording
              </button>
              <p className="record-note">Max {MAX_RECORD_SECONDS} seconds. Answer the on-screen questions.</p>
            </div>
          )}

          {recording && (
            <div>
              <div className="recording-indicator">
                <span className="rec-dot"></span> Recording...
              </div>
              <button className="btn btn-danger" onClick={stopRecording}>
                Stop Recording
              </button>
            </div>
          )}

          {recordedBlob && (
            <div>
              <p className="record-done">Recording complete!</p>
              <button className="btn btn-primary" onClick={uploadRecording}>
                Upload & Analyze
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setRecordedBlob(null)}
              >
                Discard
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
