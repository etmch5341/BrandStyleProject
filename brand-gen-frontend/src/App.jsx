import { useState, useRef, useCallback } from "react";

// ─── Config ────────────────────────────────────────────────────────────────
const API_BASE = "https://agc9y5fncg.execute-api.us-east-2.amazonaws.com/prod";
const API_KEY  = "KRdjm4Xtqe8rNgfAstgQt90d86ObBksB7iQ0lEgF";
const headers  = () => ({ "X-API-Key": API_KEY });

// ─── Helpers ─────────────────────────────────────────────────────────────────
function useFileSlot() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const ref                   = useRef();

  const pick  = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };
  const clear = () => { setFile(null); setPreview(null); if (ref.current) ref.current.value = ""; };
  return { file, preview, pick, clear, ref };
}

const SCORE_LABELS = {
  reference_anchoring:       "Reference Anchoring",
  scene_specificity:         "Scene Specificity",
  photographic_language:     "Photographic Language",
  subject_clarity:           "Subject Clarity",
  style_coherence:           "Style Coherence",
  brand_commercial_readiness:"Brand Readiness",
};

function scoreColor(n) {
  if (n >= 8) return "#10b981";
  if (n >= 5) return "#f59e0b";
  return "#ef4444";
}

function deltaColor(n) {
  if (n > 0) return "#10b981";
  if (n < 0) return "#ef4444";
  return "#475569";
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function ImageSlot({ label, sub, slot }) {
  return (
    <div style={s.slot}>
      <div style={s.slotLabel}>{label}</div>
      {sub && <div style={s.slotSub}>{sub}</div>}
      <input ref={slot.ref} type="file" accept="image/*" onChange={slot.pick} style={{ display: "none" }} />
      {slot.preview ? (
        <div style={s.slotPreviewWrap}>
          <img src={slot.preview} alt={label} style={s.slotPreview} />
          <button onClick={slot.clear} style={s.clearBtn} title="Remove">✕</button>
        </div>
      ) : (
        <button onClick={() => slot.ref.current?.click()} style={s.uploadBtn}>
          <span style={{ fontSize: 18 }}>＋</span>
          <span style={{ fontSize: 10, color: "#4a5568" }}>Upload image</span>
        </button>
      )}
    </div>
  );
}

function Toggle({ value, onChange, label, sub }) {
  return (
    <div style={s.toggleRow} onClick={() => onChange(!value)}>
      <div>
        <div style={s.toggleLabel}>{label}</div>
        {sub && <div style={s.slotSub}>{sub}</div>}
      </div>
      <div style={{ ...s.toggleTrack, background: value ? "#6366f1" : "#1a1d2e" }}>
        <div style={{ ...s.toggleThumb, transform: value ? "translateX(20px)" : "translateX(2px)" }} />
      </div>
    </div>
  );
}

function ModelSelector({ value, onChange }) {
  const models = [
    { id: "klein",      label: "Klein",    sub: "Local 4B" },
    { id: "flux-2-pro", label: "Pro",      sub: "BFL API"  },
    { id: "flux-2-max", label: "Max",      sub: "BFL API"  },
  ];
  return (
    <div style={s.modelSelector}>
      {models.map(m => (
        <button
          key={m.id}
          onClick={() => onChange(m.id)}
          style={{
            ...s.modelBtn,
            ...(value === m.id ? s.modelBtnActive : {}),
          }}
        >
          <span style={{ fontWeight: 600, fontSize: 13 }}>{m.label}</span>
          <span style={{ fontSize: 10, color: value === m.id ? "#a5b4fc" : "#334155", marginTop: 2 }}>{m.sub}</span>
        </button>
      ))}
    </div>
  );
}

function StatusBadge({ status }) {
  const map = {
    queued:  { color: "#f59e0b", label: "Queued"      },
    running: { color: "#6366f1", label: "Generating…" },
    done:    { color: "#10b981", label: "Done"         },
    error:   { color: "#ef4444", label: "Error"        },
  };
  const st = map[status] || map.queued;
  return (
    <span style={{ ...s.badge, background: st.color + "22", color: st.color, borderColor: st.color + "44" }}>
      {status === "running" && <span style={s.pulse} />}
      {st.label}
    </span>
  );
}

// ── Prompt Improvement Panel ───────────────────────────────────────────────────
function ImprovementPanel({ data, onUsePrompt }) {
  if (!data) return null;
  const { evaluation: ev, optimization: opt } = data;

  return (
    <div style={s.improvePanel}>
      {/* Header */}
      <div style={s.improvePanelHeader}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 2, color: "#6366f1", textTransform: "uppercase" }}>
          Prompt Analysis
        </span>
        <span style={{
          fontSize: 22, fontWeight: 700,
          color: scoreColor(ev.overall_score),
          fontFamily: "'DM Mono', monospace",
        }}>
          {ev.overall_score.toFixed(1)}<span style={{ fontSize: 12, color: "#475569" }}>/10</span>
        </span>
      </div>

      {/* Score grid */}
      <div style={s.scoreGrid}>
        {Object.entries(SCORE_LABELS).map(([key, label]) => {
          const score = ev.scores[key] ?? 0;
          const delta = opt.score_delta[key] ?? 0;
          return (
            <div key={key} style={s.scoreCard}>
              <div style={{ fontSize: 10, color: "#475569", marginBottom: 4, lineHeight: 1.3 }}>{label}</div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span style={{ fontSize: 18, fontWeight: 700, color: scoreColor(score), fontFamily: "'DM Mono', monospace" }}>
                  {score}
                </span>
                <span style={{ fontSize: 10, color: "#334155" }}>/10</span>
                {delta !== 0 && (
                  <span style={{ fontSize: 10, color: deltaColor(delta), fontWeight: 600, marginLeft: "auto" }}>
                    {delta > 0 ? "+" : ""}{delta}
                  </span>
                )}
              </div>
              {/* Mini bar */}
              <div style={{ marginTop: 5, background: "#131720", borderRadius: 3, height: 3, overflow: "hidden" }}>
                <div style={{ width: `${score * 10}%`, height: "100%", background: scoreColor(score), borderRadius: 3, transition: "width 0.4s" }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Issues + Strengths */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
        {ev.issues?.length > 0 && (
          <div>
            <div style={{ fontSize: 10, fontWeight: 700, color: "#ef4444", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 6 }}>Issues</div>
            {ev.issues.map((issue, i) => (
              <div key={i} style={s.issueItem}>⚠ {issue}</div>
            ))}
          </div>
        )}
        {ev.strengths?.length > 0 && (
          <div>
            <div style={{ fontSize: 10, fontWeight: 700, color: "#10b981", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 6 }}>Strengths</div>
            {ev.strengths.map((str, i) => (
              <div key={i} style={s.strengthItem}>✓ {str}</div>
            ))}
          </div>
        )}
      </div>

      {/* Optimized Prompt */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "#6366f1", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 8 }}>
          Optimized Prompt
        </div>
        <div style={s.optimizedPromptBox}>
          {opt.optimized_prompt}
        </div>
        <button onClick={() => onUsePrompt(opt.optimized_prompt)} style={s.usePromptBtn}>
          ✦ Use This Prompt
        </button>
      </div>

      {/* Changes made */}
      {opt.changes_made?.length > 0 && (
        <details style={{ marginTop: 6 }}>
          <summary style={{ fontSize: 10, color: "#475569", cursor: "pointer", userSelect: "none", letterSpacing: 1, textTransform: "uppercase" }}>
            Changes Made ({opt.changes_made.length})
          </summary>
          <div style={{ marginTop: 8 }}>
            {opt.changes_made.map((c, i) => (
              <div key={i} style={s.changeItem}>→ {c}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

// ── Result Panel ───────────────────────────────────────────────────────────────
function ResultPanel({ job }) {
  if (!job) return null;

  const outputImages = [];
  if (job.status === "done") {
    if (job.fluxUrl) outputImages.push({ url: job.fluxUrl,      label: "FLUX Output" });
    if (job.upscaledUrl && job.upscaleApplied)
      outputImages.push({ url: job.upscaledUrl, label: "Upscaled (4×)" });
    if (job.finalUrl && job.vtonApplied)
      outputImages.push({ url: job.finalUrl,    label: "VTON Applied" });
  }

  const gridCols = outputImages.length === 1 ? "1fr"
                 : outputImages.length === 2 ? "1fr 1fr"
                 : "1fr 1fr 1fr";

  return (
    <div style={s.resultPanel}>
      <div style={s.resultHeader}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ fontSize: 12, color: "#94a3b8", fontFamily: "'DM Mono', monospace" }}>
            {job.id.slice(0, 8)}…
          </span>
          {job.model && (
            <span style={{ fontSize: 10, color: "#334155", textTransform: "uppercase", letterSpacing: 1 }}>
              {job.model}
            </span>
          )}
        </div>
        <StatusBadge status={job.status} />
      </div>

      {(job.status === "running" || job.status === "queued") ? (
        <div style={s.progressWrap}>
          <div style={s.progressTrack}>
            <div style={s.progressBar} />
          </div>
          <div style={{ fontSize: 12, color: "#475569", marginTop: 8 }}>
            {job.status === "queued" ? "Waiting for GPU…" : `Generating with ${job.model}…`}
          </div>
        </div>
      ) : job.status === "done" ? (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: gridCols, gap: 12, padding: 16 }}>
            {outputImages.map(({ url, label }) => (
              <div key={label} style={s.resultImgWrap}>
                <div style={s.imgLabel}>{label}</div>
                <img src={url} alt={label} style={s.resultImg} />
                <a href={url} target="_blank" rel="noopener noreferrer" style={s.dlBtn}>
                  ↓ Open Full Size
                </a>
              </div>
            ))}
          </div>
          {/* Upscale comparison note */}
          {job.upscaleApplied && (
            <div style={{ padding: "0 16px 12px", fontSize: 11, color: "#475569" }}>
              ✦ Real-ESRGAN 4× upscale applied — view full-size to compare
            </div>
          )}
        </div>
      ) : job.status === "error" ? (
        <div style={s.errorBox}>{job.error || "Unknown error"}</div>
      ) : null}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Form state
  const [prompt,          setPrompt]          = useState("");
  const [modelChoice,     setModelChoice]     = useState("flux-2-pro");
  const [useVton,         setUseVton]         = useState(false);
  const [useUpscale,      setUseUpscale]      = useState(false);
  const [upscaleStrength, setUpscaleStrength] = useState(0.9);
  const [garmentCategory, setGarmentCategory] = useState("tops");
  const [fluxSteps,       setFluxSteps]       = useState(28);
  const [guidanceScale,   setGuidanceScale]   = useState(3.5);
  const [vtonSteps,       setVtonSteps]       = useState(30);
  const [genWidth,        setGenWidth]        = useState(1024);
  const [genHeight,       setGenHeight]       = useState(1024);

  // Image slots
  const modelSlot   = useFileSlot();
  const bgSlot      = useFileSlot();
  const garmentSlot = useFileSlot();
  const addRef1Slot = useFileSlot();
  const addRef2Slot = useFileSlot();

  // Prompt improvement state
  const [improving,       setImproving]     = useState(false);
  const [improvement,     setImprovement]   = useState(null);
  const [improveError,    setImproveError]  = useState(null);
  const [improveOpen,     setImproveOpen]   = useState(false);

  // Job tracking
  const [jobs,         setJobs]         = useState([]);
  const [submitting,   setSubmitting]   = useState(false);
  const [activeJobId,  setActiveJobId]  = useState(null);
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
          status:        data.status,
          model:         data.model,
          fluxUrl:       data.flux_url,
          upscaledUrl:   data.upscaled_url,
          finalUrl:      data.final_url,
          upscaleApplied: data.upscale_applied,
          vtonApplied:   data.vton_applied,
          error:         data.error,
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
    fd.append("model_choice",     modelChoice);
    fd.append("use_vton",         useVton);
    fd.append("use_upscale",      useUpscale);
    fd.append("upscale_strength", upscaleStrength);
    fd.append("garment_category", garmentCategory);
    fd.append("flux_steps",       fluxSteps);
    fd.append("guidance_scale",   guidanceScale);
    fd.append("vton_steps",       vtonSteps);
    fd.append("width",            genWidth);
    fd.append("height",           genHeight);
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
        model: modelChoice, prompt,
        fluxUrl: null, upscaledUrl: null, finalUrl: null,
        upscaleApplied: false, vtonApplied: false, error: null,
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

  const handleImprovePrompt = async () => {
    if (!prompt.trim()) return;
    setImproving(true);
    setImproveError(null);
    setImprovement(null);
    setImproveOpen(true);

    const fd = new FormData();
    fd.append("prompt", prompt);

    try {
      const res  = await fetch(`${API_BASE}/improve-prompt`, { method: "POST", headers: headers(), body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Improve failed");
      setImprovement(data);
    } catch (e) {
      setImproveError(e.message);
    } finally {
      setImproving(false);
    }
  };

  const activeJob = jobs.find(j => j.id === activeJobId) || jobs[0] || null;

  const modelLabel = { klein: "FLUX.2 Klein", "flux-2-pro": "FLUX.2 Pro", "flux-2-max": "FLUX.2 Max" };

  return (
    <div style={s.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #07090f; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0f1117; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }
        @keyframes slide { 0% { transform: translateX(-100%) } 100% { transform: translateX(400%) } }
        @keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }
        details summary::-webkit-details-marker { display: none; }
        details > summary::before { content: "▸ "; font-size: 10px; }
        details[open] > summary::before { content: "▾ "; }
      `}</style>

      {/* ── Header ── */}
      <div style={s.header}>
        <div style={s.headerInner}>
          <div>
            <div style={s.headerTag}>Brand Marketing Asset Generator</div>
            <h1 style={s.headerTitle}>
              {modelLabel[modelChoice]}
              <span style={{ fontSize: 12, color: "#475569", fontWeight: 400, marginLeft: 10 }}>
                + VTON 1.5
              </span>
            </h1>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={s.dot} />
            <span style={{ fontSize: 12, color: "#475569", fontFamily: "'DM Mono', monospace" }}>EC2 GPU</span>
          </div>
        </div>
      </div>

      <div style={s.layout}>
        {/* ── LEFT PANEL ── */}
        <div style={s.leftPanel}>

          {/* Model Selector */}
          <div style={s.section}>
            <div style={s.sectionTitle}>Model</div>
            <div style={s.sectionSub}>Select inference backend</div>
            <ModelSelector value={modelChoice} onChange={setModelChoice} />
            {modelChoice === "klein" && (
              <div style={s.warnBox}>⚠ Klein uses local GPU — ensure KLEIN_ENABLED=true on EC2</div>
            )}
          </div>

          {/* Prompt */}
          <div style={s.section}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <div style={s.sectionTitle}>Prompt</div>
              <button
                onClick={handleImprovePrompt}
                disabled={improving || !prompt.trim()}
                style={{
                  ...s.improveBtn,
                  opacity: (improving || !prompt.trim()) ? 0.4 : 1,
                  cursor: (improving || !prompt.trim()) ? "not-allowed" : "pointer",
                }}
              >
                {improving ? (
                  <><span style={s.miniSpinner} /> Analyzing…</>
                ) : (
                  "✦ Improve"
                )}
              </button>
            </div>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder={`Professional urban lifestyle marketing photograph...\n\nTIP: Reference images as "image 1" (bg), "image 2" (model), etc.`}
              style={s.textarea}
              rows={5}
            />

            {/* Improvement panel */}
            {improveOpen && (
              <div style={{ marginTop: 10 }}>
                {improving && (
                  <div style={s.improvingState}>
                    <span style={s.spinner} />
                    <span style={{ fontSize: 12, color: "#6366f1" }}>
                      Ollama is evaluating your prompt…
                    </span>
                  </div>
                )}
                {improveError && (
                  <div style={s.errorBox}>{improveError}</div>
                )}
                {improvement && (
                  <ImprovementPanel
                    data={improvement}
                    onUsePrompt={(p) => { setPrompt(p); setImproveOpen(false); }}
                  />
                )}
              </div>
            )}
          </div>

          {/* Reference Images */}
          <div style={s.section}>
            <div style={s.sectionTitle}>Reference Images</div>
            <div style={s.sectionSub}>Order: Background (1) → Model (2) → Additional (3, 4)</div>
            <div style={s.slotGrid}>
              <ImageSlot label="Background / Location" sub="→ Image 1 in prompt" slot={bgSlot} />
              <ImageSlot label="Model / Person"        sub="→ Image 2 in prompt" slot={modelSlot} />
              <ImageSlot label="Additional Ref 1"      sub="→ Image 3 in prompt" slot={addRef1Slot} />
              <ImageSlot label="Additional Ref 2"      sub="→ Image 4 in prompt" slot={addRef2Slot} />
            </div>
          </div>

          {/* Garment */}
          <div style={s.section}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <div>
                <div style={s.sectionTitle}>Garment Image</div>
                <div style={s.sectionSub}>For VTON clothing consistency</div>
              </div>
              <select value={garmentCategory} onChange={e => setGarmentCategory(e.target.value)} style={s.select}>
                <option value="tops">Tops</option>
                <option value="bottoms">Bottoms</option>
                <option value="one-pieces">One-pieces</option>
              </select>
            </div>
            <div style={{ maxWidth: "50%" }}>
              <ImageSlot label="Garment" sub="Clean product shot" slot={garmentSlot} />
            </div>
          </div>

          {/* Post-Processing Toggles */}
          <div style={s.section}>
            <div style={s.sectionTitle}>Post-Processing</div>

            {/* VTON */}
            <Toggle
              value={useVton}
              onChange={setUseVton}
              label="Virtual Try-On (VTON)"
              sub="Apply Fashn VTON 1.5 for garment consistency"
            />
            {useVton && !garmentSlot.file && (
              <div style={{ ...s.warnBox, marginTop: 8 }}>⚠ Upload a garment image to use VTON</div>
            )}

            <div style={s.divider} />

            {/* Upscale */}
            <Toggle
              value={useUpscale}
              onChange={setUseUpscale}
              label="Real-ESRGAN 4× Upscale"
              sub="Enhances detail and sharpness post-generation"
            />
            {useUpscale && (
              <div style={{ marginTop: 12 }}>
                <label style={s.advLabel}>
                  Upscale Strength
                  <span style={s.advVal}>{upscaleStrength.toFixed(2)}</span>
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                  <span style={{ fontSize: 10, color: "#334155" }}>Blend</span>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={upscaleStrength}
                    onChange={e => setUpscaleStrength(+e.target.value)}
                    style={s.range}
                  />
                  <span style={{ fontSize: 10, color: "#334155" }}>Full</span>
                </div>
                <div style={s.strengthHint}>
                  {upscaleStrength < 0.3
                    ? "Light enhancement — subtle sharpening, original character preserved"
                    : upscaleStrength < 0.7
                    ? "Balanced blend — enhanced detail with natural look"
                    : "Full ESRGAN — maximum detail and sharpness enhancement"}
                </div>
              </div>
            )}
          </div>

          {/* Advanced Settings */}
          <details style={s.details}>
            <summary style={s.detailsSummary}>Advanced Settings</summary>
            <div style={s.advGrid}>
              <div style={s.advItem}>
                <label style={s.advLabel}>FLUX Steps <span style={s.advVal}>{fluxSteps}</span></label>
                <input type="range" min={1} max={50} value={fluxSteps}
                  onChange={e => setFluxSteps(+e.target.value)} style={s.range} />
                <div style={s.advHint}>{modelChoice === "klein" ? "Klein: 4 steps typical" : "Pro/Max: 28–35 steps typical"}</div>
              </div>
              <div style={s.advItem}>
                <label style={s.advLabel}>Guidance Scale <span style={s.advVal}>{guidanceScale.toFixed(1)}</span></label>
                <input type="range" min={1} max={10} step={0.5} value={guidanceScale}
                  onChange={e => setGuidanceScale(+e.target.value)} style={s.range} />
              </div>
              {modelChoice !== "klein" && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  <div style={s.advItem}>
                    <label style={s.advLabel}>Width <span style={s.advVal}>{genWidth}</span></label>
                    <input type="range" min={512} max={2048} step={64} value={genWidth}
                      onChange={e => setGenWidth(+e.target.value)} style={s.range} />
                  </div>
                  <div style={s.advItem}>
                    <label style={s.advLabel}>Height <span style={s.advVal}>{genHeight}</span></label>
                    <input type="range" min={512} max={2048} step={64} value={genHeight}
                      onChange={e => setGenHeight(+e.target.value)} style={s.range} />
                  </div>
                </div>
              )}
              {useVton && (
                <div style={s.advItem}>
                  <label style={s.advLabel}>VTON Steps <span style={s.advVal}>{vtonSteps}</span></label>
                  <input type="range" min={10} max={50} value={vtonSteps}
                    onChange={e => setVtonSteps(+e.target.value)} style={s.range} />
                </div>
              )}
            </div>
          </details>

          {/* Generate button */}
          <button
            onClick={handleSubmit}
            disabled={submitting || !prompt.trim()}
            style={{
              ...s.generateBtn,
              opacity: (submitting || !prompt.trim()) ? 0.5 : 1,
              cursor:  (submitting || !prompt.trim()) ? "not-allowed" : "pointer",
            }}
          >
            {submitting ? (
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={s.spinner} /> Submitting…
              </span>
            ) : (
              `Generate${useVton ? " + VTON" : ""}${useUpscale ? " + 4× Upscale" : ""}`
            )}
          </button>
        </div>

        {/* ── RIGHT PANEL — Results ── */}
        <div style={s.rightPanel}>
          <div style={s.sectionTitle}>Results</div>

          {jobs.length === 0 ? (
            <div style={s.emptyState}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🎨</div>
              <div style={{ fontSize: 14, color: "#334155" }}>No generations yet</div>
              <div style={{ fontSize: 12, color: "#1e293b", marginTop: 4 }}>Configure a prompt and hit Generate</div>
            </div>
          ) : (
            <div>
              <ResultPanel job={activeJob} />
              {jobs.filter(j => j.id !== activeJob?.id).length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div style={s.pastTitle}>Previous Jobs</div>
                  {jobs.filter(j => j.id !== activeJob?.id).map(job => (
                    <div key={job.id} style={s.pastItem} onClick={() => setActiveJobId(job.id)}>
                      <div style={{ fontSize: 10, color: "#475569", fontFamily: "'DM Mono', monospace" }}>
                        {job.id.slice(0, 8)}…
                      </div>
                      <div style={{ fontSize: 11, color: "#334155", marginLeft: 6, textTransform: "uppercase", letterSpacing: 0.5 }}>
                        {job.model}
                      </div>
                      <div style={{ fontSize: 12, color: "#94a3b8", flex: 1, marginLeft: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {job.prompt?.slice(0, 55)}
                      </div>
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

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = {
  root: { fontFamily: "'DM Sans', sans-serif", background: "#07090f", minHeight: "100vh", color: "#e2e8f0" },

  header: { borderBottom: "1px solid #0f1117", background: "#07090f", position: "sticky", top: 0, zIndex: 50, backdropFilter: "blur(12px)" },
  headerInner: { maxWidth: 1400, margin: "0 auto", padding: "14px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" },
  headerTag: { fontSize: 10, letterSpacing: 3, color: "#475569", textTransform: "uppercase", marginBottom: 2 },
  headerTitle: { fontSize: 20, fontWeight: 600, color: "#f1f5f9" },
  dot: { width: 8, height: 8, borderRadius: "50%", background: "#10b981", animation: "pulse 2s infinite" },

  layout: { maxWidth: 1400, margin: "0 auto", padding: "24px", display: "grid", gridTemplateColumns: "440px 1fr", gap: 24, alignItems: "start" },
  leftPanel: { display: "flex", flexDirection: "column", gap: 4 },
  rightPanel: { background: "#0b0d14", border: "1px solid #131720", borderRadius: 14, padding: 24, minHeight: 400 },

  section: { background: "#0b0d14", border: "1px solid #131720", borderRadius: 12, padding: "14px 16px", marginBottom: 6 },
  sectionTitle: { fontSize: 11, fontWeight: 700, letterSpacing: 1.8, textTransform: "uppercase", color: "#475569", marginBottom: 4 },
  sectionSub: { fontSize: 10, color: "#2d3748", marginBottom: 10 },
  divider: { height: 1, background: "#131720", margin: "12px 0" },

  // Model selector
  modelSelector: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 },
  modelBtn: {
    background: "#07090f", border: "1px solid #1a1d2e", borderRadius: 8,
    padding: "10px 8px", cursor: "pointer", display: "flex", flexDirection: "column",
    alignItems: "center", gap: 2, transition: "all 0.15s", color: "#475569",
  },
  modelBtnActive: {
    background: "#0f1030", border: "1px solid #6366f1",
    color: "#e0e7ff", boxShadow: "0 0 12px rgba(99,102,241,0.2)",
  },

  textarea: {
    width: "100%", background: "#07090f", border: "1px solid #1e293b", borderRadius: 8,
    color: "#cbd5e1", padding: "10px 12px", fontSize: 13, lineHeight: 1.6,
    resize: "vertical", outline: "none", fontFamily: "'DM Sans', sans-serif",
  },

  // Improve button (inline)
  improveBtn: {
    background: "transparent", border: "1px solid #6366f144", borderRadius: 6,
    color: "#818cf8", fontSize: 11, fontWeight: 600, padding: "4px 10px",
    cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
    letterSpacing: 0.3, transition: "all 0.15s", fontFamily: "'DM Sans', sans-serif",
  },
  miniSpinner: {
    display: "inline-block", width: 10, height: 10,
    border: "1.5px solid rgba(129,140,248,0.3)", borderTopColor: "#818cf8",
    borderRadius: "50%", animation: "spin 0.7s linear infinite",
  },
  improvingState: {
    display: "flex", alignItems: "center", gap: 10, padding: "12px",
    background: "#0a0c1a", border: "1px solid #6366f122", borderRadius: 8,
  },

  // Improvement panel
  improvePanel: {
    background: "#08091a", border: "1px solid #1e2040",
    borderRadius: 10, padding: "14px", marginTop: 0,
  },
  improvePanelHeader: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    marginBottom: 14, paddingBottom: 10, borderBottom: "1px solid #131720",
  },
  scoreGrid: {
    display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 14,
  },
  scoreCard: {
    background: "#0b0d20", border: "1px solid #1a1d35", borderRadius: 8, padding: "10px 10px 8px",
  },
  issueItem: {
    fontSize: 11, color: "#fca5a5", background: "#ef444411", border: "1px solid #ef444422",
    borderRadius: 5, padding: "5px 8px", marginBottom: 4, lineHeight: 1.4,
  },
  strengthItem: {
    fontSize: 11, color: "#86efac", background: "#10b98111", border: "1px solid #10b98122",
    borderRadius: 5, padding: "5px 8px", marginBottom: 4, lineHeight: 1.4,
  },
  optimizedPromptBox: {
    background: "#07090f", border: "1px solid #6366f133", borderRadius: 8,
    padding: "10px 12px", fontSize: 12, color: "#c7d2fe", lineHeight: 1.6,
    marginBottom: 8,
  },
  usePromptBtn: {
    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    border: "none", borderRadius: 7, color: "#fff",
    padding: "8px 14px", fontSize: 12, fontWeight: 600,
    cursor: "pointer", width: "100%", fontFamily: "'DM Sans', sans-serif",
  },
  changeItem: {
    fontSize: 11, color: "#64748b", padding: "4px 0", borderBottom: "1px solid #0f1117",
    lineHeight: 1.4,
  },

  // Image slots
  slotGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 },
  slot: { display: "flex", flexDirection: "column", gap: 4 },
  slotLabel: { fontSize: 11, fontWeight: 500, color: "#94a3b8" },
  slotSub: { fontSize: 10, color: "#2d3748" },
  uploadBtn: {
    background: "#07090f", border: "1.5px dashed #1e293b", borderRadius: 8,
    color: "#334155", padding: "16px 8px", cursor: "pointer",
    display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
    width: "100%",
  },
  slotPreviewWrap: { position: "relative", borderRadius: 8, overflow: "hidden", lineHeight: 0 },
  slotPreview: { width: "100%", height: 86, objectFit: "cover", borderRadius: 8, display: "block" },
  clearBtn: {
    position: "absolute", top: 4, right: 4,
    background: "rgba(0,0,0,0.7)", border: "none", color: "#fff",
    borderRadius: "50%", width: 20, height: 20, fontSize: 10,
    cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
  },

  toggleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", userSelect: "none" },
  toggleLabel: { fontSize: 13, fontWeight: 500, color: "#cbd5e1" },
  toggleTrack: { width: 44, height: 24, borderRadius: 12, position: "relative", transition: "background 0.2s", flexShrink: 0 },
  toggleThumb: { position: "absolute", top: 2, width: 20, height: 20, background: "#fff", borderRadius: "50%", transition: "transform 0.2s" },

  strengthHint: { fontSize: 10, color: "#334155", marginTop: 5, lineHeight: 1.5 },

  warnBox: { background: "#f59e0b11", border: "1px solid #f59e0b44", borderRadius: 6, padding: "6px 10px", fontSize: 10, color: "#f59e0b" },

  details: { background: "#0b0d14", border: "1px solid #131720", borderRadius: 12, padding: "12px 16px", marginBottom: 6 },
  detailsSummary: { fontSize: 11, fontWeight: 700, letterSpacing: 1.8, textTransform: "uppercase", color: "#475569", cursor: "pointer", userSelect: "none" },
  advGrid: { display: "flex", flexDirection: "column", gap: 12, marginTop: 12 },
  advItem: { display: "flex", flexDirection: "column", gap: 4 },
  advLabel: { fontSize: 11, color: "#64748b", display: "flex", justifyContent: "space-between" },
  advVal: { color: "#6366f1", fontFamily: "'DM Mono', monospace" },
  advHint: { fontSize: 10, color: "#2d3748" },
  range: { width: "100%", accentColor: "#6366f1" },
  select: { background: "#07090f", border: "1px solid #1e293b", borderRadius: 6, color: "#cbd5e1", padding: "5px 8px", fontSize: 12, outline: "none" },

  generateBtn: {
    width: "100%", background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    border: "none", borderRadius: 10, color: "#fff", padding: "14px",
    fontSize: 14, fontWeight: 600, letterSpacing: 0.5,
    fontFamily: "'DM Sans', sans-serif", transition: "opacity 0.15s", marginBottom: 4,
  },
  spinner: { width: 14, height: 14, border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", animation: "spin 0.8s linear infinite", display: "inline-block" },

  badge: { fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 20, border: "1px solid", letterSpacing: 0.5, display: "inline-flex", alignItems: "center", gap: 5 },
  pulse: { width: 6, height: 6, borderRadius: "50%", background: "currentColor", animation: "pulse 1.2s infinite", display: "inline-block" },

  resultPanel: { background: "#07090f", border: "1px solid #131720", borderRadius: 12, overflow: "hidden", marginBottom: 12 },
  resultHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid #131720" },
  progressWrap: { padding: "24px 16px", textAlign: "center" },
  progressTrack: { background: "#1e293b", borderRadius: 4, height: 4, overflow: "hidden" },
  progressBar: { height: "100%", width: "40%", background: "linear-gradient(90deg, #6366f1, #8b5cf6)", borderRadius: 4, animation: "slide 1.5s ease-in-out infinite" },
  resultImgWrap: { display: "flex", flexDirection: "column", gap: 6 },
  imgLabel: { fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: 1.2, textTransform: "uppercase" },
  resultImg: { width: "100%", borderRadius: 8, display: "block" },
  dlBtn: { fontSize: 11, color: "#6366f1", textAlign: "center", textDecoration: "none", padding: "4px 0", display: "block" },

  errorBox: { padding: 14, color: "#ef4444", fontSize: 12, background: "#ef444411", borderRadius: 6, margin: 12 },

  emptyState: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 320 },
  pastTitle: { fontSize: 10, letterSpacing: 2, color: "#2d3748", textTransform: "uppercase", marginBottom: 6 },
  pastItem: { display: "flex", alignItems: "center", padding: "9px 12px", borderRadius: 8, background: "#07090f", border: "1px solid #131720", marginBottom: 5, cursor: "pointer", gap: 6 },
};