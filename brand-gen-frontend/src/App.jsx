import { useState, useRef, useCallback } from "react";

// ─── Config ────────────────────────────────────────────────────────────────
// Replace with your actual API Gateway URL and key
const API_BASE = "https://agc9y5fncg.execute-api.us-east-2.amazonaws.com/prod";
const API_KEY  = "KRdjm4Xtqe8rNgfAstgQt90d86ObBksB7iQ0lEgF";

const headers = () => ({ "X-API-Key": API_KEY });

// ─── Helpers ────────────────────────────────────────────────────────────────
function useFileSlot() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const ref = useRef();

  const pick = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };
  const clear = () => { setFile(null); setPreview(null); if (ref.current) ref.current.value = ""; };

  return { file, preview, pick, clear, ref };
}

const handleDownload = async (url, filename) => {
  const response = await fetch(url);
  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(blobUrl);
};

// ─── Sub-components ─────────────────────────────────────────────────────────
function ImageSlot({ label, sub, slot, required = false }) {
  return (
    <div style={styles.slot}>
      <div style={styles.slotLabel}>
        {label}
        {required && <span style={styles.required}>*</span>}
      </div>
      {sub && <div style={styles.slotSub}>{sub}</div>}

      <input
        ref={slot.ref} type="file" accept="image/*"
        onChange={slot.pick} style={{ display: "none" }}
      />

      {slot.preview ? (
        <div style={styles.slotPreviewWrap}>
          <img src={slot.preview} alt={label} style={styles.slotPreview} />
          <button onClick={slot.clear} style={styles.clearBtn} title="Remove">✕</button>
        </div>
      ) : (
        <button onClick={() => slot.ref.current?.click()} style={styles.uploadBtn}>
          <span style={{ fontSize: 20 }}>＋</span>
          <span style={{ fontSize: 11, color: "#64748b" }}>Upload image</span>
        </button>
      )}
    </div>
  );
}

function Toggle({ value, onChange, label, sub }) {
  return (
    <div style={styles.toggleRow} onClick={() => onChange(!value)}>
      <div>
        <div style={styles.toggleLabel}>{label}</div>
        {sub && <div style={styles.slotSub}>{sub}</div>}
      </div>
      <div style={{ ...styles.toggleTrack, background: value ? "#6366f1" : "#1e293b" }}>
        <div style={{ ...styles.toggleThumb, transform: value ? "translateX(20px)" : "translateX(2px)" }} />
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const map = {
    queued:  { color: "#f59e0b", label: "Queued" },
    running: { color: "#6366f1", label: "Generating…" },
    done:    { color: "#10b981", label: "Done" },
    error:   { color: "#ef4444", label: "Error" },
  };
  const s = map[status] || map.queued;
  return (
    <span style={{ ...styles.badge, background: s.color + "22", color: s.color, borderColor: s.color + "44" }}>
      {status === "running" && <span style={styles.pulse} />}
      {s.label}
    </span>
  );
}

function ResultPanel({ job }) {
  if (!job) return null;
  return (
    <div style={styles.resultPanel}>
      <div style={styles.resultHeader}>
        <span style={{ fontSize: 13, color: "#94a3b8" }}>Job {job.id.slice(0, 8)}…</span>
        <StatusBadge status={job.status} />
      </div>

      {job.status === "running" || job.status === "queued" ? (
        <div style={styles.progressWrap}>
          <div style={styles.progressTrack}>
            <div style={styles.progressBar} />
          </div>
          <div style={{ fontSize: 12, color: "#475569", marginTop: 8 }}>
            {job.status === "queued" ? "Waiting for GPU…" : "FLUX generating" + (job.vton ? " → VTON applying" : "") + "…"}
          </div>
        </div>
      ) : job.status === "done" ? (
        <div>
          <div style={styles.resultImages}>
            {job.fluxUrl && (
              <div style={styles.resultImgWrap}>
                <div style={styles.imgLabel}>FLUX Output</div>
                <img src={job.fluxUrl} alt="Flux output" style={styles.resultImg} />
                {/* <button 
                  onClick={() => handleDownload(job.fluxUrl, "flux_output.png")} 
                  style={{...styles.dlBtn, background: "none", border: "none", cursor: "pointer"}}
                >
                  ↓ Download
                </button> */}
                <a href={job.fluxUrl} target="_blank" rel="noopener noreferrer" style={styles.dlBtn}>↓ Open Image</a>
              </div>
            )}
            {job.finalUrl && job.vtonApplied && (
              <div style={styles.resultImgWrap}>
                <div style={styles.imgLabel}>Final (VTON Applied)</div>
                <img src={job.finalUrl} alt="VTON output" style={styles.resultImg} />
                {/* <button 
                  onClick={() => handleDownload(job.finalUrl, "final_output.png")} 
                  style={{...styles.dlBtn, background: "none", border: "none", cursor: "pointer"}}
                >
                  ↓ Download
                </button> */}
                <a href={job.finalUrl} target="_blank" rel="noopener noreferrer" style={styles.dlBtn}>↓ Open Image</a>
              </div>
            )}
          </div>
        </div>
      ) : job.status === "error" ? (
        <div style={styles.errorBox}>{job.error || "Unknown error"}</div>
      ) : null}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  // Form state
  const [prompt,          setPrompt]          = useState("");
  const [useVton,         setUseVton]         = useState(false);
  const [garmentCategory, setGarmentCategory] = useState("tops");
  const [fluxSteps,       setFluxSteps]       = useState(4);
  const [guidanceScale,   setGuidanceScale]   = useState(4.0);
  const [vtonSteps,       setVtonSteps]       = useState(30);

  // Image slots
  const modelSlot      = useFileSlot();
  const bgSlot         = useFileSlot();
  const garmentSlot    = useFileSlot();
  const addRef1Slot    = useFileSlot();
  const addRef2Slot    = useFileSlot();

  // Job tracking
  const [jobs,           setJobs]           = useState([]);
  const [submitting,     setSubmitting]     = useState(false);
  const [activeJobId,    setActiveJobId]    = useState(null);
  const pollRef = useRef({});

  const updateJob = useCallback((id, data) => {
    setJobs(prev => prev.map(j => j.id === id ? { ...j, ...data } : j));
  }, []);

  const pollJob = useCallback((jobId) => {
    const interval = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE}/status/${jobId}`, { headers: headers() });
        const data = await res.json();

        updateJob(jobId, {
          status:     data.status,
          fluxUrl:    data.flux_url,
          finalUrl:   data.final_url,
          vtonApplied: data.vton_applied,
          error:      data.error,
        });

        if (data.status === "done" || data.status === "error") {
          clearInterval(interval);
          delete pollRef.current[jobId];
        }
      } catch (e) {
        console.error("Poll error:", e);
      }
    }, 3000);

    pollRef.current[jobId] = interval;
  }, [updateJob]);

  const handleSubmit = async () => {
    if (!prompt.trim()) return;
    setSubmitting(true);

    const fd = new FormData();
    fd.append("prompt",           prompt);
    fd.append("use_vton",         useVton);
    fd.append("garment_category", garmentCategory);
    fd.append("flux_steps",       fluxSteps);
    fd.append("guidance_scale",   guidanceScale);
    fd.append("vton_steps",       vtonSteps);
    if (modelSlot.file)   fd.append("model_reference",      modelSlot.file);
    if (bgSlot.file)      fd.append("background_reference", bgSlot.file);
    if (garmentSlot.file) fd.append("garment_image",        garmentSlot.file);
    if (addRef1Slot.file) fd.append("additional_ref_1",     addRef1Slot.file);
    if (addRef2Slot.file) fd.append("additional_ref_2",     addRef2Slot.file);

    try {
      const res  = await fetch(`${API_BASE}/generate`, { method: "POST", headers: headers(), body: fd });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || "Request failed");

      const newJob = {
        id: data.job_id, status: "queued",
        prompt, vton: useVton,
        fluxUrl: null, finalUrl: null, vtonApplied: false, error: null,
      };
      setJobs(prev => [newJob, ...prev]);
      setActiveJobId(data.job_id);
      pollJob(data.job_id);
    } catch (e) {
      alert("Submit failed: " + e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const activeJob = jobs.find(j => j.id === activeJobId) || jobs[0] || null;

  return (
    <div style={styles.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #07090f; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0f1117; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
        @keyframes slide { 0% { transform: translateX(-100%) } 100% { transform: translateX(400%) } }
      `}</style>

      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerInner}>
          <div>
            <div style={styles.headerTag}>FLUX.2 Klein + VTON 1.5</div>
            <h1 style={styles.headerTitle}>Brand Style Generator</h1>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={styles.dot} />
            <span style={{ fontSize: 12, color: "#475569", fontFamily: "'DM Mono', monospace" }}>EC2 GPU</span>
          </div>
        </div>
      </div>

      <div style={styles.layout}>
        {/* ── LEFT PANEL — Controls ── */}
        <div style={styles.leftPanel}>

          {/* Prompt */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Prompt</div>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder={`Professional urban lifestyle marketing photograph...\n\nTIP: Reference images as "image 1" (background), "image 2" (model), etc.`}
              style={styles.textarea}
              rows={5}
            />
          </div>

          {/* Reference Images */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Reference Images</div>
            <div style={styles.sectionSub}>Image order: Background → Model → Additional</div>
            <div style={styles.slotGrid}>
              <ImageSlot label="Background / Location" sub="→ Image 1 in prompt" slot={bgSlot} />
              <ImageSlot label="Model / Person"        sub="→ Image 2 in prompt" slot={modelSlot} />
              <ImageSlot label="Additional Ref 1"      sub="→ Image 3 in prompt" slot={addRef1Slot} />
              <ImageSlot label="Additional Ref 2"      sub="→ Image 4 in prompt" slot={addRef2Slot} />
            </div>
          </div>

          {/* Garment */}
          <div style={styles.section}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
              <div>
                <div style={styles.sectionTitle}>Garment Image</div>
                <div style={styles.sectionSub}>Used for VTON clothing consistency</div>
              </div>
              <select
                value={garmentCategory}
                onChange={e => setGarmentCategory(e.target.value)}
                style={styles.select}
              >
                <option value="tops">Tops</option>
                <option value="bottoms">Bottoms</option>
                <option value="one-pieces">One-pieces</option>
              </select>
            </div>
            <div style={styles.slotGrid}>
              <ImageSlot label="Garment" sub="Clean product shot recommended" slot={garmentSlot} />
            </div>
          </div>

          {/* VTON Toggle */}
          <div style={styles.section}>
            <Toggle
              value={useVton}
              onChange={setUseVton}
              label="Virtual Try-On (VTON)"
              sub="Apply Fashn VTON 1.5 for garment consistency after FLUX generation"
            />
            {useVton && !garmentSlot.file && (
              <div style={styles.warnBox}>⚠ Upload a garment image above to use VTON</div>
            )}
          </div>

          {/* Advanced Settings */}
          <details style={styles.details}>
            <summary style={styles.detailsSummary}>Advanced Settings</summary>
            <div style={styles.advGrid}>
              <div style={styles.advItem}>
                <label style={styles.advLabel}>FLUX Steps <span style={styles.advVal}>{fluxSteps}</span></label>
                <input type="range" min={1} max={20} value={fluxSteps} onChange={e => setFluxSteps(+e.target.value)} style={styles.range} />
              </div>
              <div style={styles.advItem}>
                <label style={styles.advLabel}>Guidance Scale <span style={styles.advVal}>{guidanceScale.toFixed(1)}</span></label>
                <input type="range" min={1} max={10} step={0.5} value={guidanceScale} onChange={e => setGuidanceScale(+e.target.value)} style={styles.range} />
              </div>
              {useVton && (
                <div style={styles.advItem}>
                  <label style={styles.advLabel}>VTON Steps <span style={styles.advVal}>{vtonSteps}</span></label>
                  <input type="range" min={10} max={50} value={vtonSteps} onChange={e => setVtonSteps(+e.target.value)} style={styles.range} />
                </div>
              )}
            </div>
          </details>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={submitting || !prompt.trim()}
            style={{
              ...styles.generateBtn,
              opacity: (submitting || !prompt.trim()) ? 0.5 : 1,
              cursor:  (submitting || !prompt.trim()) ? "not-allowed" : "pointer",
            }}
          >
            {submitting ? (
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={styles.spinner} /> Submitting…
              </span>
            ) : (
              `Generate${useVton ? " + VTON" : ""}`
            )}
          </button>
        </div>

        {/* ── RIGHT PANEL — Results ── */}
        <div style={styles.rightPanel}>
          <div style={styles.sectionTitle} >Results</div>

          {jobs.length === 0 ? (
            <div style={styles.emptyState}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🎨</div>
              <div style={{ fontSize: 14, color: "#334155" }}>No generations yet</div>
              <div style={{ fontSize: 12, color: "#1e293b", marginTop: 4 }}>Fill in a prompt and hit Generate</div>
            </div>
          ) : (
            <div>
              {/* Active job — large */}
              <ResultPanel job={activeJob} />

              {/* Past jobs — list */}
              {jobs.filter(j => j.id !== activeJob?.id).length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div style={styles.pastTitle}>Previous Jobs</div>
                  {jobs.filter(j => j.id !== activeJob?.id).map(job => (
                    <div key={job.id} style={styles.pastItem} onClick={() => setActiveJobId(job.id)}>
                      <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'DM Mono', monospace" }}>{job.id.slice(0, 8)}…</div>
                      <div style={{ fontSize: 12, color: "#94a3b8", flex: 1, marginLeft: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{job.prompt.slice(0, 60)}</div>
                      <StatusBadge status={job.status} />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const styles = {
  root: {
    fontFamily: "'DM Sans', sans-serif",
    background: "#07090f",
    minHeight: "100vh",
    color: "#e2e8f0",
  },
  header: {
    borderBottom: "1px solid #0f1117",
    background: "#07090f",
    position: "sticky", top: 0, zIndex: 50,
    backdropFilter: "blur(12px)",
  },
  headerInner: {
    maxWidth: 1280, margin: "0 auto", padding: "14px 24px",
    display: "flex", alignItems: "center", justifyContent: "space-between",
  },
  headerTag: { fontSize: 10, letterSpacing: 3, color: "#475569", textTransform: "uppercase", marginBottom: 2 },
  headerTitle: { fontSize: 20, fontWeight: 600, color: "#f1f5f9" },
  dot: { width: 8, height: 8, borderRadius: "50%", background: "#10b981", animation: "pulse 2s infinite" },

  layout: {
    maxWidth: 1280, margin: "0 auto", padding: "24px",
    display: "grid", gridTemplateColumns: "420px 1fr", gap: 24,
    alignItems: "start",
  },

  leftPanel: { display: "flex", flexDirection: "column", gap: 4 },
  rightPanel: { background: "#0b0d14", border: "1px solid #131720", borderRadius: 14, padding: 24, minHeight: 400 },

  section: { background: "#0b0d14", border: "1px solid #131720", borderRadius: 12, padding: "16px 18px", marginBottom: 8 },
  sectionTitle: { fontSize: 12, fontWeight: 600, letterSpacing: 1.5, textTransform: "uppercase", color: "#64748b", marginBottom: 6 },
  sectionSub: { fontSize: 11, color: "#334155", marginBottom: 12 },

  textarea: {
    width: "100%", background: "#07090f", border: "1px solid #1e293b", borderRadius: 8,
    color: "#cbd5e1", padding: "10px 12px", fontSize: 13, lineHeight: 1.6,
    resize: "vertical", outline: "none", fontFamily: "'DM Sans', sans-serif",
  },

  slotGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 },
  slot: { display: "flex", flexDirection: "column", gap: 4 },
  slotLabel: { fontSize: 11, fontWeight: 500, color: "#94a3b8" },
  slotSub: { fontSize: 10, color: "#334155" },
  required: { color: "#6366f1", marginLeft: 3 },

  uploadBtn: {
    background: "#07090f", border: "1.5px dashed #1e293b", borderRadius: 8,
    color: "#334155", padding: "18px 8px", cursor: "pointer",
    display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
    width: "100%", transition: "border-color 0.15s",
  },
  slotPreviewWrap: { position: "relative", borderRadius: 8, overflow: "hidden", lineHeight: 0 },
  slotPreview: { width: "100%", height: 90, objectFit: "cover", borderRadius: 8, display: "block" },
  clearBtn: {
    position: "absolute", top: 4, right: 4,
    background: "rgba(0,0,0,0.7)", border: "none", color: "#fff",
    borderRadius: "50%", width: 20, height: 20, fontSize: 10,
    cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
  },

  toggleRow: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    cursor: "pointer", userSelect: "none",
  },
  toggleLabel: { fontSize: 13, fontWeight: 500, color: "#cbd5e1" },
  toggleTrack: { width: 44, height: 24, borderRadius: 12, position: "relative", transition: "background 0.2s", flexShrink: 0 },
  toggleThumb: { position: "absolute", top: 2, width: 20, height: 20, background: "#fff", borderRadius: "50%", transition: "transform 0.2s" },

  warnBox: { background: "#f59e0b11", border: "1px solid #f59e0b44", borderRadius: 6, padding: "8px 10px", fontSize: 11, color: "#f59e0b", marginTop: 10 },

  details: { background: "#0b0d14", border: "1px solid #131720", borderRadius: 12, padding: "14px 18px", marginBottom: 8 },
  detailsSummary: { fontSize: 12, fontWeight: 600, letterSpacing: 1.5, textTransform: "uppercase", color: "#475569", cursor: "pointer", userSelect: "none" },
  advGrid: { display: "flex", flexDirection: "column", gap: 12, marginTop: 14 },
  advItem: { display: "flex", flexDirection: "column", gap: 6 },
  advLabel: { fontSize: 12, color: "#64748b", display: "flex", justifyContent: "space-between" },
  advVal: { color: "#6366f1", fontFamily: "'DM Mono', monospace" },
  range: { width: "100%", accentColor: "#6366f1" },
  select: { background: "#07090f", border: "1px solid #1e293b", borderRadius: 6, color: "#cbd5e1", padding: "5px 8px", fontSize: 12, outline: "none" },

  generateBtn: {
    width: "100%", background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    border: "none", borderRadius: 10, color: "#fff",
    padding: "14px", fontSize: 14, fontWeight: 600, letterSpacing: 0.5,
    fontFamily: "'DM Sans', sans-serif", transition: "opacity 0.15s",
    marginBottom: 4,
  },
  spinner: { width: 14, height: 14, border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", animation: "spin 0.8s linear infinite", display: "inline-block" },

  badge: { fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 20, border: "1px solid", letterSpacing: 0.5, display: "inline-flex", alignItems: "center", gap: 5 },
  pulse: { width: 6, height: 6, borderRadius: "50%", background: "currentColor", animation: "pulse 1.2s infinite", display: "inline-block" },

  resultPanel: { background: "#07090f", border: "1px solid #131720", borderRadius: 12, overflow: "hidden" },
  resultHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid #131720" },

  progressWrap: { padding: "24px 16px", textAlign: "center" },
  progressTrack: { background: "#1e293b", borderRadius: 4, height: 4, overflow: "hidden" },
  progressBar: { height: "100%", width: "40%", background: "linear-gradient(90deg, #6366f1, #8b5cf6)", borderRadius: 4, animation: "slide 1.5s ease-in-out infinite" },

  resultImages: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, padding: 16 },
  resultImgWrap: { display: "flex", flexDirection: "column", gap: 6 },
  imgLabel: { fontSize: 10, color: "#475569", fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" },
  resultImg: { width: "100%", borderRadius: 8, display: "block" },
  dlBtn: { fontSize: 11, color: "#6366f1", textAlign: "center", textDecoration: "none", padding: "4px 0" },

  errorBox: { padding: 16, color: "#ef4444", fontSize: 13, background: "#ef444411" },

  emptyState: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 320 },
  pastTitle: { fontSize: 11, letterSpacing: 2, color: "#334155", textTransform: "uppercase", marginBottom: 8 },
  pastItem: { display: "flex", alignItems: "center", padding: "10px 12px", borderRadius: 8, background: "#07090f", border: "1px solid #131720", marginBottom: 6, cursor: "pointer" },
};
