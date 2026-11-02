import React, { useState, useRef } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function UploadPage({ onSessionStart }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const videoRef = useRef(null);
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const fileInputRef = useRef(null);

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
        videoRef.current.srcObject = null;
      };

      recorder.start();
      setRecording(true);
    } catch (err) {
      alert("Camera access denied: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      setRecording(false);
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

          <video
            ref={videoRef}
            className="preview-video"
            muted
            style={{ display: recording ? "block" : "none" }}
          />

          {!recording && !recordedBlob && (
            <button className="btn btn-primary" onClick={startRecording}>
              Start Recording
            </button>
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
              <p>Recording ready!</p>
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
