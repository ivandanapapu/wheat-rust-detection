import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { UploadCloud, Leaf, ShieldCheck, AlertTriangle, Activity } from "lucide-react";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setResult(null);
    if (selected) setPreview(URL.createObjectURL(selected));
  };

  const predict = async () => {
    if (!file) return alert("Please upload a wheat leaf image first.");
    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "Backend not connected. Check VITE_API_URL or API deployment." });
    }
    setLoading(false);
  };

  const isRust = result?.label === "Rust Detected";

  return (
    <div className="page">
      <nav className="navbar">
        <div className="brand"><Leaf size={28}/> WheatVision AI</div>
        <span>Binary CNN Classifier</span>
      </nav>

      <section className="hero">
        <div className="heroText">
          <p className="badge">ML Project • Wheat Rust Detection</p>
          <h1>Detect Wheat Leaf Rust using Deep Learning</h1>
          <p className="sub">Upload a wheat leaf image and the CNN model predicts whether the leaf is healthy or rust affected.</p>
          <div className="stats">
            <div><Activity/><b>CNN</b><span>Image Classification</span></div>
            <div><ShieldCheck/><b>Metrics</b><span>Precision • Recall • F1</span></div>
            <div><AlertTriangle/><b>Focus</b><span>False Negative Rate</span></div>
          </div>
        </div>

        <div className="card uploadCard">
          <label className="dropBox">
            {preview ? <img src={preview} alt="preview" /> : <><UploadCloud size={48}/><p>Click to upload wheat leaf image</p><small>JPG, PNG, JPEG</small></>}
            <input type="file" accept="image/*" onChange={handleFile}/>
          </label>
          <button onClick={predict} disabled={loading}>{loading ? "Analyzing..." : "Analyze Leaf"}</button>
        </div>
      </section>

      {result && (
        <section className={`result ${isRust ? "rust" : "healthy"}`}>
          {result.error ? <h2>{result.error}</h2> : <>
            <h2>{result.label}</h2>
            <p className="confidence">Confidence: {result.confidence}%</p>
            <div className="meter"><div style={{width: `${result.confidence}%`}}></div></div>
            <p>{result.recommendation}</p>
            <p className="prob">Rust Probability: {result.rust_probability}</p>
          </>}
        </section>
      )}

      <section className="infoGrid">
        <div className="info"><h3>Data Augmentation</h3><p>Training uses flip, rotation, zoom and contrast jitter to improve model generalization.</p></div>
        <div className="info"><h3>Evaluation</h3><p>The model reports precision, recall, F1-score and confusion matrix after validation.</p></div>
        <div className="info"><h3>False Negative Check</h3><p>False negatives are rust leaves predicted as healthy. This is important because missed disease can spread.</p></div>
      </section>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
