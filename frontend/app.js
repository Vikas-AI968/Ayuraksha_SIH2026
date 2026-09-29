/**
 * IP-SAKTI Sahayak — Frontend JavaScript
 * Multilingual RAG, Web Speech API (STT), SpeechSynthesis API (TTS).
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const backendUrlInput = document.getElementById("backend-url");
  const languageSelect = document.getElementById("language-select");
  const jurisdictionSelect = document.getElementById("jurisdiction-select");
  const queryInput = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const submitSpinner = document.getElementById("submit-spinner");
  const submitLabel = document.getElementById("submit-label");
  const clearBtn = document.getElementById("clear-btn");
  const sttBtn = document.getElementById("stt-btn");
  const sttIcon = document.getElementById("stt-icon");
  const sttLabel = document.getElementById("stt-label");
  const sttStatus = document.getElementById("stt-status");
  const systemStatusDot = document.querySelector(".status-dot");
  const backendStatusText = document.getElementById("backend-status-text");

  // Results elements
  const resultsCard = document.getElementById("results-card");
  const statusBadge = document.getElementById("status-badge");
  const languageBadge = document.getElementById("language-badge");
  const jurisdictionBadge = document.getElementById("jurisdiction-badge");
  const ttsPlayBtn = document.getElementById("tts-play-btn");
  const ttsStopBtn = document.getElementById("tts-stop-btn");
  const ttsStatus = document.getElementById("tts-status");
  const answerText = document.getElementById("answer-text");
  const keyPointsSection = document.getElementById("key-points-section");
  const keyPointsList = document.getElementById("key-points-list");
  const confidenceLevel = document.getElementById("confidence-level");
  const confidenceScore = document.getElementById("confidence-score");
  const confidenceFactors = document.getElementById("confidence-factors");
  const citationCoveragePct = document.getElementById("citation-coverage-pct");
  const citationProgress = document.getElementById("citation-progress");
  const warningsSection = document.getElementById("warnings-section");
  const warningsList = document.getElementById("warnings-list");
  const citationsCount = document.getElementById("citations-count");
  const citationsContainer = document.getElementById("citations-container");
  const evidenceCount = document.getElementById("evidence-count");
  const evidenceContainer = document.getElementById("evidence-container");
  const disclaimerText = document.getElementById("disclaimer-text");

  let currentResponse = null;
  let recognition = null;
  let isListening = false;

  // Language to Speech locales mapping
  const SPEECH_LOCALES = {
    en: "en-IN",
    hi: "hi-IN",
    te: "te-IN",
    auto: "en-IN"
  };

  // 1. Check Backend Health
  async function checkBackendHealth() {
    const baseUrl = backendUrlInput.value.trim().replace(/\/+$/, "");
    try {
      const resp = await fetch(`${baseUrl}/health`, { method: "GET" });
      if (resp.ok) {
        const data = await resp.json();
        systemStatusDot.className = "status-dot online";
        backendStatusText.textContent = `Backend online (${data.indexed_chunks || 0} chunks indexed)`;
      } else {
        systemStatusDot.className = "status-dot offline";
        backendStatusText.textContent = `Backend status: ${resp.status}`;
      }
    } catch (e) {
      systemStatusDot.className = "status-dot offline";
      backendStatusText.textContent = "Backend unreachable (check URL)";
    }
  }

  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);

  // 2. Chip Clicks
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const lang = chip.getAttribute("data-lang");
      const query = chip.getAttribute("data-query");
      if (lang && languageSelect.querySelector(`option[value="${lang}"]`)) {
        languageSelect.value = lang;
      }
      queryInput.value = query;
      queryInput.focus();
    });
  });

  // 3. Clear Button
  clearBtn.addEventListener("click", () => {
    queryInput.value = "";
    resultsCard.classList.add("hidden");
    stopTTS();
    queryInput.focus();
  });

  // 4. Speech-to-Text (STT) via Web Speech API
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      isListening = true;
      sttBtn.classList.add("listening");
      sttLabel.textContent = "Listening...";
      sttStatus.textContent = `Listening in ${recognition.lang}... Speak now.`;
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      queryInput.value = (queryInput.value ? queryInput.value + " " : "") + transcript;
      sttStatus.textContent = `Transcribed: "${transcript}"`;
    };

    recognition.onerror = (event) => {
      sttStatus.textContent = `STT error: ${event.error}`;
      resetSTTButton();
    };

    recognition.onend = () => {
      resetSTTButton();
    };

    sttBtn.addEventListener("click", () => {
      if (isListening) {
        recognition.stop();
        resetSTTButton();
      } else {
        const langCode = languageSelect.value;
        recognition.lang = SPEECH_LOCALES[langCode] || "en-IN";
        try {
          recognition.start();
        } catch (e) {
          sttStatus.textContent = `Could not start speech recognition: ${e.message}`;
          resetSTTButton();
        }
      }
    });
  } else {
    sttBtn.addEventListener("click", () => {
      sttStatus.textContent = "Speech recognition (STT) is not supported in this browser.";
    });
  }

  function resetSTTButton() {
    isListening = false;
    sttBtn.classList.remove("listening");
    sttLabel.textContent = "Speak";
  }

  // 5. Text-to-Speech (TTS) via Web Speech API
  function stopTTS() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      ttsPlayBtn.classList.remove("hidden");
      ttsStopBtn.classList.add("hidden");
      ttsStatus.textContent = "";
    }
  }

  ttsStopBtn.addEventListener("click", stopTTS);

  ttsPlayBtn.addEventListener("click", () => {
    if (!("speechSynthesis" in window)) {
      ttsStatus.textContent = "Text-to-speech is not supported in this browser.";
      return;
    }

    if (!currentResponse || !currentResponse.answer) {
      return;
    }

    window.speechSynthesis.cancel(); // Cancel any ongoing speech

    const langCode = currentResponse.language || languageSelect.value || "en";
    const targetLocale = SPEECH_LOCALES[langCode] || "en-IN";

    const utterance = new SpeechSynthesisUtterance(currentResponse.answer);
    utterance.lang = targetLocale;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    // Pick matching voice if available
    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(v => v.lang === targetLocale || v.lang.startsWith(langCode));
    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onstart = () => {
      ttsPlayBtn.classList.add("hidden");
      ttsStopBtn.classList.remove("hidden");
      ttsStatus.textContent = `Speaking (${targetLocale})...`;
    };

    utterance.onend = () => {
      stopTTS();
    };

    utterance.onerror = (e) => {
      ttsStatus.textContent = `TTS error: ${e.error || "playback failed"}`;
      stopTTS();
    };

    window.speechSynthesis.speak(utterance);
  });

  // Pre-load voices if supported
  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      window.speechSynthesis.getVoices();
    };
  }

  // 6. Submit Query (RAG Pipeline)
  submitBtn.addEventListener("click", handleQuerySubmit);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleQuerySubmit();
    }
  });

  async function handleQuerySubmit() {
    const query = queryInput.value.trim();
    if (!query) {
      alert("Please enter a query.");
      queryInput.focus();
      return;
    }

    const baseUrl = backendUrlInput.value.trim().replace(/\/+$/, "");
    const language = languageSelect.value;
    const jurisdiction = jurisdictionSelect.value;

    setLoading(true);
    stopTTS();

    try {
      const response = await fetch(`${baseUrl}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          language: language,
          jurisdiction: jurisdiction,
          top_k: 5
        })
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errJson = await response.json();
          errorDetail = errJson.detail || JSON.stringify(errJson);
        } catch (_) {}
        throw new Error(errorDetail);
      }

      const data = await response.json();
      currentResponse = data;
      renderResponse(data);
    } catch (error) {
      alert(`Query failed: ${error.message}\nEnsure FastAPI backend is running at ${baseUrl}`);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    if (loading) {
      submitBtn.disabled = true;
      submitSpinner.classList.remove("hidden");
      submitLabel.textContent = "Analyzing & Retrieving...";
    } else {
      submitBtn.disabled = false;
      submitSpinner.classList.add("hidden");
      submitLabel.textContent = "Ask IP-SAKTI";
    }
  }

  // 7. Render Response
  function renderResponse(data) {
    resultsCard.classList.remove("hidden");
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });

    // Status Badge
    statusBadge.textContent = data.status.toUpperCase();
    if (data.status === "answered") {
      statusBadge.className = "badge";
    } else if (data.status === "abstained") {
      statusBadge.className = "badge badge-abstained";
    } else {
      statusBadge.className = "badge badge-outline";
    }

    // Language Badge
    const resolvedLang = data.language || "en";
    languageBadge.textContent = `Lang: ${resolvedLang.toUpperCase()}`;

    // Jurisdiction Badge
    const resolvedJurisdiction = (data.analysis && data.analysis.jurisdiction) || "India";
    jurisdictionBadge.textContent = `Jurisdiction: ${resolvedJurisdiction}`;

    // Answer Text
    answerText.textContent = data.answer || (data.abstained ? "The system abstained from answering this query based on lack of authoritative support." : "No answer returned.");

    // Key Points
    if (data.key_points && data.key_points.length > 0) {
      keyPointsSection.classList.remove("hidden");
      keyPointsList.innerHTML = "";
      data.key_points.forEach(kp => {
        const li = document.createElement("li");
        // key_points items are {point: string, evidence_ids: string[]} objects
        const pointText = (typeof kp === "string") ? kp : (kp.point || "");
        const ids = (typeof kp === "object" && kp.evidence_ids && kp.evidence_ids.length > 0)
          ? ` <span class="kp-ids">[${kp.evidence_ids.join(", ")}]</span>`
          : "";
        li.innerHTML = escapeHtml(pointText) + ids;
        keyPointsList.appendChild(li);
      });
    } else {
      keyPointsSection.classList.add("hidden");
    }

    // Confidence
    const conf = data.confidence || {};
    const score = typeof conf.score === "number" ? conf.score : 0.0;
    const level = conf.level || "low";
    confidenceScore.textContent = score.toFixed(2);
    confidenceLevel.textContent = level.toUpperCase();
    confidenceLevel.className = `confidence-tag ${level}`;

    if (conf.factors) {
      confidenceFactors.innerHTML = "";
      Object.entries(conf.factors).forEach(([k, v]) => {
        const span = document.createElement("span");
        span.textContent = `${k}: ${v}`;
        confidenceFactors.appendChild(span);
      });
    }

    // Citation Coverage
    const coverage = typeof data.citation_coverage === "number" ? data.citation_coverage : 0.0;
    const pct = Math.round(coverage * 100);
    citationCoveragePct.textContent = `${pct}%`;
    citationProgress.style.width = `${pct}%`;

    // Warnings / Notices
    const warnings = data.warnings || [];
    if (warnings.length > 0) {
      warningsSection.classList.remove("hidden");
      warningsList.innerHTML = "";
      warnings.forEach(w => {
        const li = document.createElement("li");
        li.textContent = w;
        warningsList.appendChild(li);
      });
    } else {
      warningsSection.classList.add("hidden");
    }

    // Citations
    const citations = data.citations || [];
    citationsCount.textContent = citations.length;
    citationsContainer.innerHTML = "";

    if (citations.length === 0) {
      citationsContainer.innerHTML = `<p class="citation-meta">No specific claims cited.</p>`;
    } else {
      citations.forEach((c, idx) => {
        const card = document.createElement("div");
        card.className = "citation-card";
        const supportedTag = c.supported ? "✅ Verified" : "⚠️ Uncertain";
        const evIds = (c.evidence_ids || []).join(", ") || "None";
        card.innerHTML = `
          <div class="citation-title">${idx + 1}. ${escapeHtml(c.claim)}</div>
          <div class="citation-meta">Status: <strong>${supportedTag}</strong> | Cited Evidence: [${escapeHtml(evIds)}] | Confidence: ${c.confidence || 0.0}</div>
        `;
        citationsContainer.appendChild(card);
      });
    }

    // Evidence
    const evidenceItems = data.evidence || [];
    evidenceCount.textContent = evidenceItems.length;
    evidenceContainer.innerHTML = "";

    if (evidenceItems.length === 0) {
      evidenceContainer.innerHTML = `<p class="citation-meta">No evidence items returned.</p>`;
    } else {
      evidenceItems.forEach(ev => {
        const item = document.createElement("div");
        item.className = "evidence-item";
        const sectionBadge = ev.section ? `— Section: ${escapeHtml(ev.section)}` : "";
        const urlLink = ev.source_url ? `<br><a href="${escapeHtml(ev.source_url)}" target="_blank" class="citation-url">${escapeHtml(ev.source_url)}</a>` : "";
        item.innerHTML = `
          <div class="evidence-header">
            <span>[${escapeHtml(ev.evidence_id || "")}] ${escapeHtml(ev.title || "")} ${sectionBadge}</span>
            <span class="evidence-score">Relevance: ${ev.relevance_score != null ? ev.relevance_score.toFixed(2) : "N/A"}</span>
          </div>
          <div class="citation-meta">Authority: ${escapeHtml(ev.authority || "")} | Jurisdiction: ${escapeHtml(ev.jurisdiction || "")} | Type: ${escapeHtml(ev.document_type || "")}</div>
          <div class="evidence-text">${escapeHtml(ev.text || "")}</div>
          ${urlLink}
        `;
        evidenceContainer.appendChild(item);
      });
    }

    // Disclaimer
    if (data.disclaimer) {
      disclaimerText.textContent = data.disclaimer;
    }
  }

  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
