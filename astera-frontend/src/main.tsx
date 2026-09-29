import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { gsap } from "gsap";
import Lenis from "lenis";
import {
  ArrowUpRight,
  Menu,
  X,
  MousePointer2,
  Mic,
  ArrowUp,
  LoaderCircle,
  Sparkles,
  ExternalLink,
  ChevronDown,
  BarChart3,
  CheckCircle2,
  ShieldCheck,
} from "lucide-react";
import "./styles.css";

type Citation = {
  id?: string;
  citation?: string;
  title?: string;
  source?: string;
  url?: string;
  document_id?: string;
  [key: string]: unknown;
};

type Evidence = {
  evidence_id?: string;
  chunk_id?: string;
  document_id?: string;
  source_id?: string;
  official_url?: string;
  source_url?: string;
  authority?: string;
  jurisdiction?: string;
  domain?: string;
  document_type?: string;
  title?: string;
  section?: string;
  subsection?: string;
  page?: number;
  text?: string;
  relevance_score?: number;
  rerank_score?: number;
  authority_score?: number;
  source_priority?: number;
  authority_level?: string;
  status?: string;
  provenance?: string;
  retrieval_method?: string;
  citation?: string;
  source_type?: string;
  access_status?: string;
  effective_date?: string;
  version?: string;
  [key: string]: unknown;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  evidence?: Evidence[];
  confidence?: number;
  status?: string;
  metrics?: { retrieval?: number; citation?: number; faithfulness?: number };
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

const more = [
  [
    "Retrieval Hit Rate/Recall@k",
    "Measures how effectively the system retrieves relevant information from the knowledge base.",
  ],
  [
    "Citation Correctness",
    "Validates whether the provided citations are accurate, relevant, and properly attributed.",
  ],
  [
    "Faithfulness",
    "Ensures the response is grounded in the retrieved content and free from hallucinations.",
  ],
];

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatAnswer(text: string) {
  return text.split("\n").map((line, index) => (
    <React.Fragment key={`${index}-${line}`}>
      {line}
      {index < text.split("\n").length - 1 && <br />}
    </React.Fragment>
  ));
}

function Scene() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
    camera.position.set(0, 0, 11);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    ref.current.appendChild(renderer.domElement);

    const N = 6500,
      geo = new THREE.BufferGeometry(),
      pos = new Float32Array(N * 3);
    const from = new Float32Array(N * 3),
      to = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const k = i * 3,
        u = i / N,
        a = u * Math.PI * 18 + (Math.random() - 0.5) * 0.42;
      const r = 0.3 + 6.7 * Math.pow(u, 0.56),
        band = (Math.random() - 0.5) * 0.65;
      from[k] = (Math.random() - 0.5) * 18;
      from[k + 1] = (Math.random() - 0.5) * 12;
      from[k + 2] = (Math.random() - 0.5) * 10;
      to[k] = Math.cos(a) * r + Math.cos(a * 3) * band;
      to[k + 1] = Math.sin(a) * r * 0.58 + Math.sin(a * 2) * band;
      to[k + 2] = (Math.random() - 0.5) * 2.8 + Math.sin(a * 1.7) * 0.25;
      pos[k] = from[k];
      pos[k + 1] = from[k + 1];
      pos[k + 2] = from[k + 2];
    }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x8fa78a,
      size: 0.018,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    });
    const cloud = new THREE.Points(geo, mat);
    scene.add(cloud);

    const ringGeo = new THREE.TorusGeometry(3.1, 0.012, 8, 160);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xb79a55, transparent: true, opacity: 0.11 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = 0.72;
    scene.add(ring);

    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.7, 3),
      new THREE.MeshBasicMaterial({ color: 0x5d8066, transparent: true, opacity: 0.045, wireframe: true }),
    );
    scene.add(core);

    let scroll = 0,
      mx = 0,
      my = 0,
      raf = 0;
    const move = (e: MouseEvent) => {
      mx = e.clientX / innerWidth - 0.5;
      my = e.clientY / innerHeight - 0.5;
    };
    const scr = () => {
      const m = document.documentElement.scrollHeight - innerHeight;
      scroll = Math.max(0, Math.min(1, scrollY / Math.max(1, m)));
    };
    const resize = () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    };
    addEventListener("mousemove", move);
    addEventListener("scroll", scr, { passive: true });
    addEventListener("resize", resize);
    scr();
    const clock = new THREE.Clock();
    const frame = () => {
      const t = clock.getElapsedTime(),
        a = geo.attributes.position.array as Float32Array;
      const p = gsap.utils.interpolate(0, 1, Math.min(1, scroll * 1.35));
      for (let i = 0; i < N; i++) {
        const k = i * 3,
          wob = Math.sin(t * 0.55 + i * 0.021) * 0.022;
        a[k] = from[k] + (to[k] - from[k]) * p + wob;
        a[k + 1] = from[k + 1] + (to[k + 1] - from[k + 1]) * p + Math.cos(t * 0.42 + i * 0.014) * 0.018;
        a[k + 2] = from[k + 2] + (to[k + 2] - from[k + 2]) * p;
      }
      geo.attributes.position.needsUpdate = true;
      cloud.rotation.z = t * 0.012 + scroll * 0.55;
      cloud.rotation.y += (mx * 0.15 - cloud.rotation.y) * 0.025;
      cloud.rotation.x += (-my * 0.1 - cloud.rotation.x) * 0.025;
      ring.rotation.z = t * 0.12 + scroll * 2.2;
      ring.rotation.y = 0.72 + my * 0.2;
      ring.scale.setScalar(0.9 + scroll * 0.42);
      core.rotation.x = t * 0.08;
      core.rotation.y = t * 0.12;
      camera.position.x += (mx * 0.65 - camera.position.x) * 0.025;
      camera.position.y += (-my * 0.35 - camera.position.y) * 0.025;
      camera.position.z += (11 - scroll * 1.9 - camera.position.z) * 0.025;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(frame);
    };
    frame();
    return () => {
      cancelAnimationFrame(raf);
      removeEventListener("mousemove", move);
      removeEventListener("scroll", scr);
      removeEventListener("resize", resize);
      geo.dispose();
      mat.dispose();
      ringGeo.dispose();
      ringMat.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);
  return <div className="scene" ref={ref} />;
}

function AssistantChat({ onMetrics }: { onMetrics: (metrics: { retrieval?: number; citation?: number; faithfulness?: number }) => void }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [jurisdiction, setJurisdiction] = useState<"India" | "International">("India");
  const [language, setLanguage] = useState<"en" | "hi" | "te">("en");
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<any>(null);
  const sessionIdRef = useRef(makeId());

  const sendMessage = async (forcedText?: string) => {
    const query = (forcedText ?? input).trim();
    if (!query || loading) return;

    setInput("");
    setMessages((current) => [...current, { id: makeId(), role: "user", content: query }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          session_id: sessionIdRef.current,
          language,
          jurisdiction,
          top_k: 5,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || `Backend request failed (${response.status})`);
      }

      // The backend exposes retrieval strength plus claim-level citation validation.
      // Keep Citation Correctness and Faithfulness distinct instead of displaying
      // the same citation_coverage value for both.
      const claimValidations = Array.isArray(data?.citations) ? data.citations : [];
      const claimConfidences = claimValidations
        .map((claim: any) => claim?.confidence)
        .filter((value: unknown): value is number => typeof value === "number" && Number.isFinite(value));
      const supportedClaims = claimValidations.filter((claim: any) => claim?.supported === true).length;
      const citationCorrectness = claimConfidences.length
        ? claimConfidences.reduce((sum: number, value: number) => sum + value, 0) / claimConfidences.length
        : undefined;
      const faithfulness = claimValidations.length
        ? supportedClaims / claimValidations.length
        : (typeof data?.citation_coverage === "number" ? data.citation_coverage : undefined);

      const m = data?.metrics || {};
      const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
      const liveMetrics = {
        retrieval: num(m.retrieval) ?? (typeof data?.confidence?.factors?.retrieval === "number" ? data.confidence.factors.retrieval : undefined),
        citation: num(m.citation_correctness) ?? citationCorrectness,
        faithfulness: num(m.faithfulness) ?? faithfulness,
      };
      onMetrics(liveMetrics);

      setMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: data.answer || "The backend returned no answer.",
          citations: Array.isArray(data.citations) ? data.citations : [],
          evidence: Array.isArray(data.evidence) ? data.evidence : [],
          confidence: typeof data?.confidence?.score === "number" ? data.confidence.score : undefined,
          status: data.status,
          metrics: liveMetrics,
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to reach the backend.";
      setMessages((current) => [
        ...current,
        {
          id: makeId(),
          role: "assistant",
          content: `I couldn't connect to the IP-SAKTI backend. ${message}`,
          status: "error",
        },
      ]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  };

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value);
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`;
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  const startVoice = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }
    const recognition = new SpeechRecognition();
    const speechLocales: Record<"en" | "hi" | "te", string> = { en: "en-IN", hi: "hi-IN", te: "te-IN" };
    recognition.lang = speechLocales[language];
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);
    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) transcript += event.results[i][0].transcript;
      setInput(transcript);
    };
    recognitionRef.current = recognition;
    recognition.start();
  };

  const clearChat = () => {
    setMessages([]);
    sessionIdRef.current = makeId();
    setInput("");
    setExpandedSources({});
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  return (
    <div className={`chatSection ${messages.length ? "hasMessages" : ""}`}>
      {messages.length === 0 ? (
        <div className="chatIntro">
          <div className="chatOrb"><Sparkles size={17} /><span className="orbLeaf">❧</span></div>
          <h2>Ask the knowledge<br /><em>within reach.</em></h2>
          <p>Explore Ayurveda and traditional knowledge, understand IP pathways, and trace every answer back to evidence.</p>
        </div>
      ) : (
        <div className="conversation">
          <div className="conversationTop">
            <span>IP-SAKTI / SAHAYAK</span>
            <button type="button" onClick={clearChat}>New chat</button>
          </div>
          <div className="messageList">
            {messages.map((message) => (
              <div className={`messageRow ${message.role}`} key={message.id}>
                <div className="messageLabel">{message.role === "user" ? "YOU" : "SAHAYAK"}</div>
                <div className="messageBody">
                  <div className="messageContent">{formatAnswer(message.content)}</div>
                  {message.confidence !== undefined && (
                    <div className="confidenceLine">Confidence {Math.round(message.confidence * 100)}%</div>
                  )}
                  {message.metrics && (
                    <div className="liveMetrics">
                      <div className="liveMetricsHead"><span>QUERY METRICS</span><span>LIVE</span></div>
                      <div className="liveMetricGrid">
                        <div><span>Retrieval</span><strong>{message.metrics.retrieval !== undefined ? `${Math.round(message.metrics.retrieval * 100)}%` : "—"}</strong></div>
                        <div><span>Citation correctness</span><strong>{message.metrics.citation !== undefined ? `${Math.round(message.metrics.citation * 100)}%` : "—"}</strong></div>
                        <div><span>Faithfulness</span><strong>{message.metrics.faithfulness !== undefined ? `${Math.round(message.metrics.faithfulness * 100)}%` : "—"}</strong></div>
                      </div>
                    </div>
                  )}
                  {!!message.evidence?.length && (
                    <div className="citations">
                      <div className="citationHeading">Sources · {message.evidence.length}</div>
                      {message.evidence.slice(0, 8).map((source, index) => {
                        const key = `${message.id}-${source.document_id || source.chunk_id || index}`;
                        const expanded = !!expandedSources[key];
                        const title = source.title || source.citation || source.document_id || `Source ${index + 1}`;
                        const sourceUrl = source.official_url || source.source_url;
                        const score = typeof source.rerank_score === "number" ? source.rerank_score : source.relevance_score;
                        return (
                          <div className={`citation sourceCard ${expanded ? "expanded" : ""}`} key={key}>
                            <button type="button" className="sourceHeader" onClick={() => setExpandedSources((current) => ({ ...current, [key]: !current[key] }))}>
                              <span className="sourceNumber">{String(index + 1).padStart(2, "0")}</span>
                              <span className="sourceTitle">{title}</span>
                              <ChevronDown size={14} className="sourceChevron" />
                            </button>
                            {expanded && (
                              <div className="sourceDetails">
                                <div className="sourceMeta">
                                  {source.authority && <span>{source.authority}</span>}
                                  {source.jurisdiction && <span>{source.jurisdiction}</span>}
                                  {source.document_type && <span>{source.document_type}</span>}
                                  {source.effective_date && <span>{source.effective_date}</span>}
                                </div>
                                {(source.citation || source.section || source.subsection || source.page) && (
                                  <p className="sourceCitation">
                                    {source.citation || ""}
                                    {source.section ? ` · ${source.section}` : ""}
                                    {source.subsection ? ` · ${source.subsection}` : ""}
                                    {source.page ? ` · p. ${source.page}` : ""}
                                  </p>
                                )}
                                {source.text && <div className="sourceExcerpt">{source.text}</div>}
                                {typeof score === "number" && <div className="sourceScore">{source.rerank_score !== undefined ? "Rerank score" : "Relevance score"} · {score.toFixed(3)}</div>}
                                {sourceUrl && (
                                  <a href={sourceUrl} target="_blank" rel="noreferrer" className="sourceLink">Open source <ExternalLink size={11} /></a>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="messageRow assistant">
                <div className="messageLabel">SAHAYAK</div>
                <div className="messageBody typing"><LoaderCircle size={15} className="spin" /> Retrieving evidence and reasoning…</div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="askBox">
        <textarea
          ref={textareaRef}
          aria-label="Ask anything"
          placeholder="Ask anything"
          rows={1}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <div className="askActions">
          <div className="languageSelectWrap">
            <label htmlFor="answer-language">LANG</label>
            <select id="answer-language" value={language} onChange={(event) => setLanguage(event.target.value as "en" | "hi" | "te")} aria-label="Answer language">
              <option value="en">English</option>
              <option value="hi">हिन्दी</option>
              <option value="te">తెలుగు</option>
            </select>
          </div>
          <div className="jurisdictionToggle" role="group" aria-label="Jurisdiction">
            <button type="button" className={jurisdiction === "India" ? "active" : ""} onClick={() => setJurisdiction("India")} aria-pressed={jurisdiction === "India"}>IND</button>
            <button type="button" className={jurisdiction === "International" ? "active" : ""} onClick={() => setJurisdiction("International")} aria-pressed={jurisdiction === "International"}>International</button>
          </div>
          <div className="askRight">
            <button
              type="button"
              aria-label="Voice input"
              title="Voice input"
              className={listening ? "listening" : ""}
              onClick={startVoice}
            ><Mic size={18} /></button>
            <button
              className="sendAsk"
              type="button"
              aria-label="Send message"
              onClick={() => void sendMessage()}
              disabled={!input.trim() || loading}
            >{loading ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={18} />}</button>
          </div>
        </div>
      </div>

      {messages.length === 0 ? (
        <div className="suggestions">
          <button type="button" onClick={() => void sendMessage("What is the patentability of a traditional Ayurvedic formulation in India?")}>Patentability</button>
          <button type="button" onClick={() => void sendMessage("Explain Section 3(p) of the Indian Patents Act.")}>Explain a section</button>
          <button type="button" onClick={() => void sendMessage("What protection is available for traditional knowledge in India?")}>Traditional knowledge</button>
        </div>
      ) : (
        <div className="chatMeta">Press Enter to send · Shift + Enter for a new line · Backend: {API_BASE}</div>
      )}
      <div className="chatNote">AI can make mistakes. Verify important legal information against authoritative sources.</div>
    </div>
  );
}

function MetricPreview({ index, metrics, onClose }: { index: number; metrics: { retrieval?: number; citation?: number; faithfulness?: number }; onClose: () => void }) {
  const metric = more[index];
  const icons = [<BarChart3 size={18} />, <CheckCircle2 size={18} />, <ShieldCheck size={18} />];
  const details = [
    "Live retrieval strength from the backend evidence pack. Benchmark Recall@k is reported separately when a gold set is available.",
    "Live citation coverage from the backend citation validator: the share of substantive claims supported by retrieved evidence with valid provenance.",
    "Live grounding rate from the same claim-level validation: the share of substantive answer claims supported by retrieved evidence."
  ];
  const values = [metrics.retrieval, metrics.citation, metrics.faithfulness];
  const value = values[index];
  return <div className="metricOverlay" onClick={onClose}>
    <div className="metricPreview" onClick={(e) => e.stopPropagation()}>
      <button className="metricClose" type="button" onClick={onClose}>×</button>
      <div className="metricPreviewIcon">{icons[index]}</div>
      <div className="eyebrow">METRIC {String(index + 1).padStart(2, "0")}</div>
      <h3>{metric[0]}</h3>
      <div className="metricPreviewValue">{value !== undefined ? `${Math.round(value * 100)}%` : "—"}</div>
      <p>{details[index]}</p>
      <div className="metricPreviewNote">Latest value from the most recent backend query. It updates when you ask Astera again.</div>
    </div>
  </div>;
}

function App() {
  const [open, setOpen] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<{ retrieval?: number; citation?: number; faithfulness?: number }>({});
  const [previewMetric, setPreviewMetric] = useState<number | null>(null);

  useEffect(() => {
    const lenis = new Lenis({ duration: 1.05, smoothWheel: true });
    let raf = 0;
    const loop = (t: number) => {
      lenis.raf(t);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    gsap.from('.nav > *', { y: -12, opacity: 0, duration: 0.7, stagger: 0.06, ease: 'power3.out' });
    gsap.from('.heroText > *', { y: 24, opacity: 0, duration: 0.85, stagger: 0.08, delay: 0.1, ease: 'power3.out' });
    return () => {
      cancelAnimationFrame(raf);
      lenis.destroy();
    };
  }, []);

  return (
    <div className="site">
      <Scene />
      <div className="ambient ambientOne" />
      <div className="ambient ambientTwo" />
      <header className="nav">
        <a className="logo" href="#top" aria-label="IP-SAKTI home">
          <span className="logoMark"><span>❋</span></span>
          <span>IP-SAKTI</span><span className="logoSub">SAHAYAK</span>
        </a>
        <nav className={open ? 'links show' : 'links'}>
          <a href="#about" onClick={() => setOpen(false)}>Purpose</a>
          <a href="#stack" onClick={() => setOpen(false)}>Sahayak</a>
          <a href="#more" onClick={() => setOpen(false)}>Trust</a>
          <a className="pill" href="#stack" onClick={() => setOpen(false)}>Ask Sahayak <ArrowUpRight size={14} /></a>
        </nav>
        <button className="menu" aria-label="Toggle navigation" onClick={() => setOpen(!open)}>{open ? <X /> : <Menu />}</button>
      </header>

      <div className="grain" />
      <main id="top">
        <section className="hero">
          <div className="heroInner">
            <div className="heroText">
              <div className="kicker"><span>01</span> AYURVEDA <i /> TRADITIONAL KNOWLEDGE <i /> INTELLIGENCE</div>
              <div className="heroTitleRow">
                <div>
                  <div className="heroOverline">INDIA'S KNOWLEDGE · MADE TRACEABLE</div>
                  <h1>IP-<em>SAKTI</em></h1>
                </div>
                <div className="heroSeal" aria-hidden="true">
                  <div className="sealRings"><span /><span /><span /></div>
                  <strong>ॐ</strong>
                  <small>ज्ञान · प्रमाण · संरक्षण</small>
                </div>
              </div>
              <p className="heroLead">A multilingual knowledge assistant for Ayurveda, traditional knowledge and intellectual property — grounded in evidence you can inspect.</p>
              <div className="heroActions">
                <a className="primaryAction" href="#stack">Ask Sahayak <ArrowUpRight size={15} /></a>
                <a className="secondaryAction" href="#about">How it works <span>↓</span></a>
              </div>
              <div className="heroTrust"><span><CheckCircle2 size={13} /> Citation-grounded</span><span><ShieldCheck size={13} /> Evidence first</span><span>EN · HI · TE</span></div>
            </div>
            <div className="heroBotanical" aria-hidden="true">
              <div className="botanicalHalo" />
              <div className="leaf leafA" /><div className="leaf leafB" /><div className="leaf leafC" /><div className="leaf leafD" />
              <div className="botanicalStem" />
              <div className="botanicalDot d1" /><div className="botanicalDot d2" /><div className="botanicalDot d3" />
            </div>
          </div>
          <div className="heroBottom"><span>IP-SAKTI / AYURVEDA KNOWLEDGE PLATFORM</span><span>SCROLL TO EXPLORE ↓</span></div>
        </section>

        <section className="about section" id="about">
          <div className="sectionHead"><div className="eyebrow">THE PURPOSE <span>01</span></div><span className="sectionKicker">ANCIENT KNOWLEDGE · MODERN CLARITY</span></div>
          <div className="split">
            <div className="purposeTitle"><span>Protect</span><strong>what India</strong><em>already knows.</em></div>
            <div className="copy">
              <div className="rule" />
              <p>Traditional knowledge can be difficult to discover, interpret and protect across modern intellectual-property systems. IP-SAKTI brings those worlds together.</p>
              <p>Ask questions in natural language, choose a jurisdiction and trace the answer to retrieved evidence and source provenance.</p>
              <a className="textLink" href="#stack">Enter the knowledge garden <ArrowUpRight size={14} /></a>
            </div>
          </div>
          <div className="principles">
            <article><span>01</span><h3>Discover</h3><p>Find relevant Ayurveda and traditional-knowledge material.</p></article>
            <article><span>02</span><h3>Understand</h3><p>Turn complex IP and regulatory language into clear guidance.</p></article>
            <article><span>03</span><h3>Verify</h3><p>Inspect citations, evidence and confidence before acting.</p></article>
          </div>
        </section>

        <section className="stack section" id="stack">
          <div className="sectionHead"><div className="eyebrow">SAHAYAK · AI KNOWLEDGE ASSISTANT <span>02</span></div><span className="sectionKicker">ASK · RETRIEVE · VERIFY</span></div>
          <div className="assistantShell">
            <div className="assistantRail">
              <div className="assistantBadge"><Sparkles size={15} /><span>SAHAYAK</span></div>
              <h2>Ask about<br /><em>Ayurveda.</em></h2>
              <p>From formulations and traditional knowledge to patents and Section 3(p), explore with evidence at the centre.</p>
              <div className="railNote"><span className="liveDot" /> KNOWLEDGE SYSTEM ONLINE</div>
            </div>
            <div className="assistantMain">
              <AssistantChat onMetrics={setLiveMetrics} />
            </div>
          </div>
        </section>

        <section className="manifesto">
          <div className="manifestoPattern" aria-hidden="true"><span /><span /><span /><span /></div>
          <div className="manifestoNo">03 · KNOWLEDGE SHOULD BE VISIBLE</div>
          <div className="manifestoCopy"><span>FROM</span><h2>ROOTS</h2><span>TO</span><h2 className="accent">REASON.</h2></div>
          <p>Ancient practices deserve modern tools that respect provenance, context and the people who carry the knowledge forward.</p>
          <div className="manifestoMeta"><span>AYURVEDA</span><span>TRADITIONAL KNOWLEDGE</span><span>INTELLECTUAL PROPERTY</span></div>
        </section>

        <section className="more section" id="more">
          <div className="sectionHead"><div className="eyebrow">TRUST & EVALUATION <span>04</span></div><span className="sectionKicker">MEASURE WHAT MATTERS</span></div>
          <div className="cards">
            {more.map((m, i) => {
              const value = [liveMetrics.retrieval, liveMetrics.citation, liveMetrics.faithfulness][i];
              const labels = ['RETRIEVAL', 'CITATION', 'GROUNDING'];
              return (
                <article className="card" key={m[0]}>
                  <div className={`art art${i}`}><span className="artIndex">0{i + 1}</span><div className="metricGlyph">{i === 0 ? '↗' : i === 1 ? '✓' : '✳'}</div><span className="artLabel">{labels[i]}</span></div>
                  <div className="cardText"><div className="cardTag">LIVE SIGNAL</div><h3>{m[0]}</h3><p>{m[1]}</p>{value !== undefined && <div className="cardLiveValue"><span>LIVE</span><strong>{Math.round(value * 100)}%</strong></div>}<button className="previewButton" type="button" onClick={() => setPreviewMetric(i)}>Understand this metric <ArrowUpRight size={14} /></button></div>
                </article>
              );
            })}
          </div>
          {previewMetric !== null && <MetricPreview index={previewMetric} metrics={liveMetrics} onClose={() => setPreviewMetric(null)} />}
        </section>

        <section className="start" id="start">
          <div className="startLeaf" aria-hidden="true">❋</div>
          <div className="eyebrow">IP-SAKTI SAHAYAK · SIH 26045</div>
          <h2>Keep the<br /><em>knowledge alive.</em></h2>
          <p>Explore Indian and international IP regimes while keeping traditional knowledge, provenance and evidence at the centre of the experience.</p>
          <a className="bigPill" href="#stack">Begin with Sahayak <ArrowUpRight size={16} /></a>
          <footer><span>IP-SAKTI · SAHAYAK</span><span>SIH 26045 · 2026</span><span>Evidence before answers</span></footer>
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
