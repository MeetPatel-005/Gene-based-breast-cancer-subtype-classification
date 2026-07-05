/* ============================================================
   BRCA Classifier — script.js
   ParticleCanvas · FileDropZone · Prediction · Charts · Demo
   ============================================================ */

// ── Subtype metadata (mirrors server SUBTYPE_INFO) ──────────────────────────
const SUBTYPE_META = {
  LumA:   { color: '#00b4d8', fullName: 'Luminal A',                 receptor: 'ER⁺ / PR⁺ / HER2⁻',      prognosis: 'Best — low proliferation, excellent long-term survival',          biomarkers: ['ESR1','PGR','GATA3','FOXA1'],          therapy: ['Endocrine therapy (Tamoxifen or Aromatase Inhibitor)', 'CDK4/6 inhibitors (Palbociclib) in advanced disease', 'Chemotherapy generally avoided if genomic risk is low'],         description: 'The most common subtype (~40% of BRCA). Driven by oestrogen/progesterone signalling. Slow-growing; responds well to hormone therapy.' },
  LumB:   { color: '#4361ee', fullName: 'Luminal B',                 receptor: 'ER⁺ / PR⁺ or ⁻ / HER2⁺ or ⁻', prognosis: 'Intermediate — higher proliferation than LumA',              biomarkers: ['ESR1','PGR','ERBB2','MKI67'],          therapy: ['Endocrine therapy + Chemotherapy', 'Anti-HER2 therapy (Trastuzumab) if HER2⁺', 'CDK4/6 inhibitors'],                                                                                   description: 'Similar to Luminal A but with higher Ki-67 proliferation. More heterogeneous; often requires combined endocrine + chemotherapy.' },
  Her2:   { color: '#f72585', fullName: 'HER2-Enriched',             receptor: 'ER⁻ / PR⁻ / HER2⁺',      prognosis: 'Intermediate-poor — aggressive but targetable',               biomarkers: ['ERBB2','GRB7','PGAP3','STARD3'],       therapy: ['Targeted anti-HER2: Trastuzumab + Pertuzumab', 'Antibody-drug conjugate: T-DM1, T-DXd', 'Chemotherapy backbone (Taxane + Carboplatin)', 'Neoadjuvant chemotherapy before surgery'],  description: 'Characterised by HER2 gene amplification (~15–20% of BRCA). Historically aggressive; dramatically improved outcomes with targeted therapy.' },
  Basal:  { color: '#ff4800', fullName: 'Basal-like (Triple-Negative)',receptor: 'ER⁻ / PR⁻ / HER2⁻',      prognosis: 'Poor — highest recurrence risk, especially within first 5 years', biomarkers: ['TP53','BRCA1','KRT5','KRT14','EGFR'], therapy: ['Chemotherapy: Anthracycline + Taxane backbone', 'Immunotherapy: Pembrolizumab (PD-L1⁺ cases)', 'PARP inhibitors (Olaparib/Talazoparib) if BRCA1/2 mutated', 'Sacituzumab govitecan (ADC) in metastatic disease'], description: 'Most aggressive subtype (~15–20%). No hormone receptors — cannot use hormone therapy. Highly responsive to chemo but relapse risk is high.' },
  Normal: { color: '#2dc653', fullName: 'Normal-like',               receptor: 'Mixed / unclear',          prognosis: 'Generally favourable — similar to Luminal A',                 biomarkers: ['ADIPOQ','DCN','PDPN'],                 therapy: ['Often treated similarly to Luminal A', 'Endocrine therapy if ER⁺', 'Clinical trial participation recommended'],                                                                           description: 'Rare subtype (~5%) that resembles normal breast tissue expression. May reflect tumour purity or adipose contamination.' },
};

// ── Utility ─────────────────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

function lerp(a, b, t) { return a + (b - a) * t; }

// ── Particle Canvas ──────────────────────────────────────────────────────────
class ParticleCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.particles = [];
    this.mouse = { x: -9999, y: -9999 };
    this.resize();
    this.init();
    this.animate();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('mousemove', e => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
  }

  resize() {
    this.canvas.width  = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    this.particles = [];
    const count = Math.floor((window.innerWidth * window.innerHeight) / 14000);
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x:   Math.random() * this.canvas.width,
        y:   Math.random() * this.canvas.height,
        vx:  (Math.random() - 0.5) * 0.35,
        vy:  (Math.random() - 0.5) * 0.35,
        r:   Math.random() * 2 + 1,
        hue: Math.random() > 0.5 ? 250 : 290, // purple / violet
      });
    }
  }

  draw() {
    const { ctx, canvas, particles, mouse } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update + draw particles
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      // Soft repulsion from mouse
      const dx = p.x - mouse.x;
      const dy = p.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 90) {
        p.x += (dx / dist) * 0.8;
        p.y += (dy / dist) * 0.8;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue}, 70%, 70%, 0.55)`;
      ctx.fill();
    }

    // Draw connecting lines between close particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 140 * 140) {
          const alpha = (1 - d2 / (140 * 140)) * 0.18;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(124, 106, 247, ${alpha})`;
          ctx.lineWidth   = 1;
          ctx.stroke();
        }
      }
    }
  }

  animate() {
    this.draw();
    requestAnimationFrame(() => this.animate());
  }
}

// ── File Drop Zone ───────────────────────────────────────────────────────────
class FileDropZone {
  constructor(zoneEl, inputEl) {
    this.zone  = zoneEl;
    this.input = inputEl;
    this.file  = null;

    this.zone.addEventListener('click', () => this.input.click());
    this.input.addEventListener('change', () => this._onFile(this.input.files[0]));

    this.zone.addEventListener('dragover',  e => { e.preventDefault(); this.zone.classList.add('drag-over'); });
    this.zone.addEventListener('dragleave', () => this.zone.classList.remove('drag-over'));
    this.zone.addEventListener('drop', e => {
      e.preventDefault();
      this.zone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) this._onFile(f);
    });
  }

  _onFile(f) {
    if (!f) return;
    this.file = f;
    $('fileName').textContent    = f.name;
    $('filePreview').style.display = 'block';
    $('dropZone').style.display    = 'none';
  }

  clear() {
    this.file = null;
    this.input.value = '';
    $('filePreview').style.display = 'none';
    $('dropZone').style.display    = '';
  }

  getFile() { return this.file; }
}

// ── Loading helpers ──────────────────────────────────────────────────────────
const LOADING_STEPS = [
  'Preparing data…',
  'Running base learners…',
  'Computing meta-predictions…',
  'Fetching clinical evidence…',
  'Building output…',
];

let _loadingInterval = null;

function showLoading() {
  $('loadingOverlay').style.display = 'flex';
  let step = 0;
  $('loadingStep').textContent = LOADING_STEPS[0];
  _loadingInterval = setInterval(() => {
    step = (step + 1) % LOADING_STEPS.length;
    $('loadingStep').textContent = LOADING_STEPS[step];
  }, 800);
}

function hideLoading() {
  clearInterval(_loadingInterval);
  $('loadingOverlay').style.display = 'none';
}

// ── Core prediction handler ──────────────────────────────────────────────────
function renderResults(data) {
  // 1. Show panel
  const panel = $('resultsPanel');
  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const subtype = data.subtype;
  const pct     = Math.round(data.confidence * 100);
  const meta    = SUBTYPE_META[subtype] || {};
  const color   = meta.color || '#7c6af7';

  // Store context for chat widget
  window._lastPrediction = data;

  // 2. Confidence ring
  renderConfidenceRing(data.confidence, color);
  $('ringPct').textContent = pct + '%';

  // 3. Subtype badge
  const badge = $('subtypeBadge');
  badge.textContent = subtype;
  badge.style.background    = `${color}22`;
  badge.style.borderColor   = `${color}88`;
  badge.style.color         = color;

  $('subtypeFullname').textContent = meta.fullName  || '';
  $('subtypeReceptor').textContent = meta.receptor  || '';
  $('rowsProcessed').textContent   = data.rows_processed > 1
    ? `${data.rows_processed} samples · showing first result`
    : '1 sample processed';

  // 4. Probability bar chart
  renderProbChart(data.probabilities, data.classes, subtype);

  // 5. Detail card
  renderDetailCard(subtype, meta, color);

  // 5.5 Top Gene Drivers panel
  renderTopGeneDrivers(data, color);

  // 6. Treatment Evidence panel
  renderEvidencePanel(data);

  // 7. Sample preview table
  if (data.preview && data.preview.length > 0) {
    renderPreviewTable(data.preview, data.preview_cols);
    $('tableCard').style.display = 'block';
  } else {
    $('tableCard').style.display = 'none';
  }
}

// ── Confidence Ring ──────────────────────────────────────────────────────────
function renderConfidenceRing(confidence, color) {
  const ring       = $('ringFill');
  const circumference = 2 * Math.PI * 50; // r=50 → 314.16
  const offset     = circumference * (1 - confidence);
  ring.style.stroke            = color;
  ring.style.strokeDasharray   = circumference;
  // Animate: start at full offset (0%), then set target
  ring.style.strokeDashoffset  = circumference;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ring.style.strokeDashoffset = offset;
    });
  });
}

// ── Probability Bar Chart ────────────────────────────────────────────────────
function renderProbChart(probs, classes, topSubtype) {
  const container = $('probChart');
  container.innerHTML = '';

  // Sort by probability descending
  const sorted = [...classes].sort((a, b) => (probs[b] || 0) - (probs[a] || 0));

  for (const cls of sorted) {
    const pct   = Math.round((probs[cls] || 0) * 100 * 10) / 10;
    const meta  = SUBTYPE_META[cls] || {};
    const color = meta.color || '#7c6af7';
    const isTop = cls === topSubtype;

    const row = document.createElement('div');
    row.className = 'prob-bar-row';
    row.innerHTML = `
      <span class="prob-bar-label" style="color:${isTop ? color : ''}">${cls}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill"
             data-pct="${probs[cls] || 0}"
             style="background:${isTop
               ? `linear-gradient(90deg, ${color}, ${color}cc)`
               : `${color}66`};
                    width:0%">
        </div>
      </div>
      <span class="prob-bar-pct">${pct}%</span>
    `;
    container.appendChild(row);
  }

  // Animate bars after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll('.prob-bar-fill').forEach(el => {
        el.style.width = (parseFloat(el.dataset.pct) * 100) + '%';
      });
    });
  });
}

// ── Detail Card ──────────────────────────────────────────────────────────────
function renderDetailCard(subtype, meta, color) {
  $('detailPrognosis').textContent = meta.prognosis || '—';

  const chipsEl = $('detailBiomarkers');
  chipsEl.innerHTML = '';
  (meta.biomarkers || []).forEach(b => {
    const chip = document.createElement('span');
    chip.className   = 'biomarker-chip';
    chip.textContent = b;
    chip.style.borderColor = `${color}55`;
    chip.style.color       = color;
    chipsEl.appendChild(chip);
  });

  const therapyEl = $('detailTherapy');
  therapyEl.innerHTML = '';
  (meta.therapy || []).forEach(t => {
    const li         = document.createElement('li');
    li.textContent   = t;
    therapyEl.appendChild(li);
  });

  $('detailDescription').textContent = meta.description || '';
}

// ── Treatment Evidence Panel ──────────────────────────────────────────────
function renderEvidencePanel(data) {
  const card = $('evidenceCard');
  const agent = data.agent_analysis || {};
  const routes = agent.therapy_routes || {};
  const warning = data.evidence_warning || agent.evidence_warning || null;

  // Nothing useful to show — hide panel
  if (!routes.primary_therapy && !warning) {
    card.style.display = 'none';
    return;
  }
  card.style.display = 'block';

  // ─ Live / offline badge ──────────────────────────────────────────
  const liveBadge = $('evidenceLiveBadge');
  if (warning) {
    liveBadge.textContent = '● Offline';
    liveBadge.className   = 'evidence-live-badge offline';
    $('evidenceWarningText').textContent = warning;
    $('evidenceWarning').style.display   = 'flex';
  } else {
    liveBadge.textContent = '● Live';
    liveBadge.className   = 'evidence-live-badge';
    $('evidenceWarning').style.display = 'none';
  }

  // ─ Build therapy list (primary + alternatives) ──────────────────────
  const listEl = $('evidenceTherapyList');
  listEl.innerHTML = '';

  const allTherapies = [];
  if (routes.primary_therapy) {
    allTherapies.push({ ...routes.primary_therapy, _isPrimary: true });
  }
  (routes.alternatives || []).forEach(t => allTherapies.push({ ...t, _isPrimary: false }));

  allTherapies.forEach(therapy => {
    const prob = therapy.probability_of_response;   // 0-1 or null
    const pct  = prob != null ? Math.round(prob * 100) : null;
    const surv = therapy['5yr_survival_rate'];
    const isPrimary = therapy._isPrimary;
    const barColor  = isPrimary
      ? 'linear-gradient(90deg, #7c6af7, #b06ef7)'
      : 'rgba(124,106,247,0.38)';

    const row = document.createElement('div');
    row.className = 'evidence-therapy-row';
    row.innerHTML = `
      <div class="evidence-therapy-header">
        <span class="evidence-therapy-name${isPrimary ? ' primary' : ''}">
          ${isPrimary ? '★ ' : ''}<span>${therapy.name || 'Unnamed therapy'}</span>
        </span>
        <div style="display:flex;align-items:center;gap:0.5rem">
          ${surv != null ? `<span class="evidence-therapy-surv">${Math.round(surv*100)}% 5yr</span>` : ''}
          ${pct  != null ? `<span class="evidence-therapy-pct">${pct}%</span>` : '<span class="evidence-therapy-pct muted">N/A</span>'}
          <span class="evidence-therapy-badge ${isPrimary ? 'primary' : 'alternative'}">
            ${isPrimary ? 'Recommended' : 'Alternative'}
          </span>
        </div>
      </div>
      ${therapy.evidence_basis ? `<div class="evidence-basis">${therapy.evidence_basis}</div>` : ''}
      <div class="evidence-therapy-track">
        <div class="evidence-therapy-fill"
             data-pct="${prob ?? 0}"
             style="background:${barColor};width:0%"></div>
      </div>
    `;
    listEl.appendChild(row);
  });

  // Animate bars
  requestAnimationFrame(() => requestAnimationFrame(() => {
    listEl.querySelectorAll('.evidence-therapy-fill').forEach(el => {
      el.style.width = (parseFloat(el.dataset.pct) * 100) + '%';
    });
  }));

  // ─ Rationale ───────────────────────────────────────────────────────────
  const rationaleEl = $('evidenceRationale');
  if (routes.rationale) {
    rationaleEl.textContent    = routes.rationale;
    rationaleEl.style.display  = 'block';
  } else {
    rationaleEl.style.display  = 'none';
  }

  // ─ Source chips ─────────────────────────────────────────────────────
  const sources = agent.evidence_sources || routes.evidence_sources || [];
  const chipsEl = $('evidenceSourceChips');
  chipsEl.innerHTML = '';

  if (sources.length) {
    sources.forEach(src => {
      const a = document.createElement('a');
      a.className = 'evidence-source-chip';
      // PubMed-searchable link
      a.href   = `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(src)}`;
      a.target = '_blank';
      a.rel    = 'noopener noreferrer';
      a.innerHTML = `🔗 ${src}`;
      chipsEl.appendChild(a);
    });
    $('evidenceSourcesSection').style.display = 'block';
  } else {
    $('evidenceSourcesSection').style.display = 'none';
  }
}

// ── KEGG Pathway Viewer Module ───────────────────────────────────────────
const keggPathway = (() => {
  // Client-side cache: gene_symbol → {pathways: [...], fetched: true} or null
  const _cache = {};

  /**
   * Toggle the KEGG pathway dropdown for a gene row.
   */
  async function toggle(geneSymbol, dropdownEl, toggleBtn) {
    const isOpen = dropdownEl.classList.contains('open');

    if (isOpen) {
      dropdownEl.classList.remove('open');
      toggleBtn.classList.remove('active');
      return;
    }

    toggleBtn.classList.add('active');
    dropdownEl.classList.add('open');

    // Already loaded?
    if (_cache[geneSymbol]) {
      return;
    }

    // Show loading state
    dropdownEl.innerHTML = `
      <div class="kegg-pathway-loading">
        <div class="kegg-pathway-spinner"></div>
        <span>Searching KEGG for ${geneSymbol} pathways…</span>
      </div>
    `;

    try {
      const resp = await fetch(`/api/kegg/pathways/${encodeURIComponent(geneSymbol)}`);
      const data = await resp.json();

      if (data.error) {
        _renderError(dropdownEl, data.error);
        _cache[geneSymbol] = { pathways: [], fetched: true };
        return;
      }

      const pathways = data.pathways || [];
      _cache[geneSymbol] = { pathways, fetched: true };

      if (pathways.length === 0) {
        _renderEmpty(dropdownEl, geneSymbol);
      } else {
        _renderPathwaySelector(dropdownEl, geneSymbol, pathways);
      }
    } catch (err) {
      _renderError(dropdownEl, 'Network error: ' + err.message);
      _cache[geneSymbol] = { pathways: [], fetched: true };
    }
  }

  /**
   * Render the "no pathways found" message.
   */
  function _renderEmpty(container, geneSymbol) {
    container.innerHTML = `
      <div class="kegg-pathway-empty">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <span>No KEGG pathway found for <strong>${geneSymbol}</strong>. This gene may not have mapped pathways in the KEGG database.</span>
      </div>
    `;
  }

  /**
   * Render the error state.
   */
  function _renderError(container, message) {
    container.innerHTML = `
      <div class="kegg-pathway-error">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01"/>
        </svg>
        <span>${message}</span>
      </div>
    `;
  }

  /**
   * Render the pathway selector dropdown + image viewer.
   */
  function _renderPathwaySelector(container, geneSymbol, pathways) {
    // Build options HTML
    const optionsHtml = pathways.map((p, i) =>
      `<option value="${p.pathway_id}"${i === 0 ? ' selected' : ''}>${p.name} (${p.pathway_id})</option>`
    ).join('');

    container.innerHTML = `
      <div class="kegg-pathway-select-wrap">
        <label class="kegg-pathway-select-label">Select Pathway (${pathways.length} found)</label>
        <select class="kegg-pathway-select" id="keggSelect_${geneSymbol}">
          ${optionsHtml}
        </select>
      </div>
      <div class="kegg-pathway-viewer" id="keggViewer_${geneSymbol}">
        <div class="kegg-img-loading">
          <div class="kegg-pathway-spinner"></div>
          <span>Loading pathway image…</span>
        </div>
      </div>
    `;

    const selectEl = container.querySelector('.kegg-pathway-select');
    const viewerEl = container.querySelector('.kegg-pathway-viewer');

    // Load the first pathway image immediately
    _loadPathwayImage(viewerEl, pathways[0].pathway_id, pathways[0].name);

    // Listen for changes
    selectEl.addEventListener('change', () => {
      const selectedId = selectEl.value;
      const selectedPathway = pathways.find(p => p.pathway_id === selectedId);
      const name = selectedPathway ? selectedPathway.name : selectedId;
      _loadPathwayImage(viewerEl, selectedId, name);
    });
  }

  /**
   * Load and display a pathway image with zoom controls.
   */
  function _loadPathwayImage(viewerEl, pathwayId, pathwayName) {
    viewerEl.innerHTML = `
      <div class="kegg-img-loading">
        <div class="kegg-pathway-spinner"></div>
        <span>Loading pathway image…</span>
      </div>
    `;

    const img = new Image();
    img.className = 'kegg-pathway-image';
    img.alt = `KEGG Pathway: ${pathwayName}`;

    img.onload = () => {
      let zoom = 100;
      const ZOOM_STEP = 20;
      const MIN_ZOOM = 40;
      const MAX_ZOOM = 200;

      viewerEl.innerHTML = '';

      // Image container (scrollable)
      const imgContainer = document.createElement('div');
      imgContainer.className = 'kegg-pathway-img-container';
      imgContainer.appendChild(img);

      // Controls bar
      const controls = document.createElement('div');
      controls.className = 'kegg-pathway-controls';
      controls.innerHTML = `
        <span class="kegg-pathway-name" title="${pathwayName}">${pathwayName}</span>
        <div class="kegg-zoom-controls">
          <button class="kegg-zoom-btn" data-action="out" title="Zoom out">−</button>
          <span class="kegg-zoom-level">${zoom}%</span>
          <button class="kegg-zoom-btn" data-action="in" title="Zoom in">+</button>
          <button class="kegg-zoom-btn" data-action="fit" title="Fit to width">⤢</button>
        </div>
      `;

      viewerEl.appendChild(imgContainer);
      viewerEl.appendChild(controls);

      const zoomLabel = controls.querySelector('.kegg-zoom-level');

      function applyZoom(newZoom) {
        zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
        img.style.transform = `scale(${zoom / 100})`;
        zoomLabel.textContent = `${zoom}%`;
      }

      controls.addEventListener('click', e => {
        const btn = e.target.closest('.kegg-zoom-btn');
        if (!btn) return;
        const action = btn.dataset.action;
        if (action === 'in')  applyZoom(zoom + ZOOM_STEP);
        if (action === 'out') applyZoom(zoom - ZOOM_STEP);
        if (action === 'fit') {
          const containerW = imgContainer.clientWidth - 16; // minus padding
          const imgW = img.naturalWidth;
          if (imgW > 0) {
            applyZoom(Math.round((containerW / imgW) * 100));
          }
        }
      });

      // Auto-fit if image is wider than container
      requestAnimationFrame(() => {
        const containerW = imgContainer.clientWidth - 16;
        if (img.naturalWidth > containerW) {
          applyZoom(Math.round((containerW / img.naturalWidth) * 100));
        }
      });
    };

    img.onerror = () => {
      viewerEl.innerHTML = `
        <div class="kegg-pathway-error">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01"/>
          </svg>
          <span>Failed to load pathway image for ${pathwayId}</span>
        </div>
      `;
    };

    // Use our proxy endpoint to avoid CORS
    img.src = `/api/kegg/pathway-image/${encodeURIComponent(pathwayId)}`;
  }

  return { toggle };
})();


// ── Top Gene Drivers Panel ───────────────────────────────────────────────
function renderTopGeneDrivers(data, accentColor) {
  const card = $('geneDriversCard');
  const drivers = data.top_gene_drivers || [];
  const geneAnalysis = (data.agent_analysis || {}).gene_analysis || {};
  const annotations = geneAnalysis.gene_annotations || {};
  const summary = geneAnalysis.summary || '';

  if (drivers.length === 0) {
    card.style.display = 'none';
    return;
  }
  card.style.display = 'block';

  $('geneDriversCount').textContent = `${drivers.length} gene${drivers.length !== 1 ? 's' : ''}`;

  const listEl = $('geneDriversList');
  listEl.innerHTML = '';

  drivers.forEach((gene, idx) => {
    const pct = Math.round(gene.importance_pct * 100);
    const exprClass = gene.expression_level === 'High' ? 'high'
                    : gene.expression_level === 'Low'  ? 'low'
                    : 'normal';
    const annotation = annotations[gene.gene_symbol] || '';
    const delay = idx * 40;

    const row = document.createElement('div');
    row.className = 'gene-driver-row';
    row.style.animationDelay = `${delay}ms`;
    row.innerHTML = `
      <div class="gene-driver-rank">#${gene.rank}</div>
      <div class="gene-driver-info">
        <div class="gene-driver-top">
          <span class="gene-driver-symbol">${gene.gene_symbol}</span>
          <span class="gene-driver-name">${gene.gene_name || gene.ensembl_id}</span>
          <button class="kegg-pathway-toggle" data-gene="${gene.gene_symbol}" title="View KEGG pathways for ${gene.gene_symbol}">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
            KEGG Pathways
          </button>
        </div>
        <div class="gene-driver-bar-track">
          <div class="gene-driver-bar-fill"
               data-pct="${gene.importance_pct}"
               style="background: linear-gradient(90deg, ${accentColor}, ${accentColor}99); width: 0%"></div>
        </div>
        ${annotation ? `<div class="gene-driver-annotation">${annotation}</div>` : ''}
      </div>
      <div class="gene-driver-meta">
        <span class="gene-importance-chip">${pct}%</span>
        <span class="gene-expression-badge ${exprClass}">${gene.expression_level}</span>
        <span class="gene-expression-value">${gene.expression_value} TPM</span>
      </div>
      <div class="kegg-pathway-dropdown" id="keggDropdown_${gene.gene_symbol}"></div>
    `;

    // Wire up the toggle button
    const toggleBtn = row.querySelector('.kegg-pathway-toggle');
    const dropdownEl = row.querySelector('.kegg-pathway-dropdown');
    toggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      keggPathway.toggle(gene.gene_symbol, dropdownEl, toggleBtn);
    });

    listEl.appendChild(row);
  });

  // Animate bars
  requestAnimationFrame(() => requestAnimationFrame(() => {
    listEl.querySelectorAll('.gene-driver-bar-fill').forEach(el => {
      el.style.width = (parseFloat(el.dataset.pct) * 100) + '%';
    });
  }));

  // LLM summary
  const summaryEl = $('geneDriversSummary');
  if (summary) {
    summaryEl.textContent = summary;
    summaryEl.style.display = 'block';
  } else {
    summaryEl.style.display = 'none';
  }
}

// ── Preview Table ────────────────────────────────────────────────────────────
function renderPreviewTable(rows, cols) {
  const table = $('previewTable');
  table.innerHTML = '';

  const thead = document.createElement('thead');
  const hr    = document.createElement('tr');
  cols.forEach(c => {
    const th = document.createElement('th');
    th.textContent = c;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach(row => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td      = document.createElement('td');
      td.textContent = row[c] !== undefined ? row[c] : '—';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

// ── Upload & Predict ─────────────────────────────────────────────────────────
async function uploadAndPredict() {
  const file = dropZone.getFile();
  if (!file) {
    flashError('Please select or drop a CSV file first.');
    return;
  }

  showLoading();
  $('resultsPanel').style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    hideLoading();

    if (data.error) {
      flashError(data.error);
    } else {
      renderResults(data);
    }
  } catch (err) {
    hideLoading();
    flashError('Network error: ' + err.message);
  }
}

// ── Demo Mode ────────────────────────────────────────────────────────────────
async function loadDemoList() {
  try {
    const resp  = await fetch('/api/demo-list');
    const data  = await resp.json();
    const files = data.files || [];

    $('demoStatus').style.display = 'none';

    if (files.length === 0) {
      $('demoEmpty').style.display = '';
    } else {
      renderDemoGrid(files);
    }
  } catch {
    $('demoDot').className     = 'status-dot error';
    $('demoStatusText').textContent = 'Could not load demo list.';
  }
}

function renderDemoGrid(files) {
  const grid = $('demoGrid');
  grid.innerHTML = '';

  files.forEach(name => {
    // Guess subtype from filename (e.g. "luma_sample" → "LumA")
    const subtypeKey = guessSubtype(name);
    const meta  = subtypeKey ? SUBTYPE_META[subtypeKey] : null;
    const color = meta ? meta.color : '#7c6af7';
    const label = meta ? meta.fullName : name;

    const btn = document.createElement('button');
    btn.className = 'demo-btn';
    btn.style.setProperty('--btn-color', color);
    btn.innerHTML = `
      <div class="demo-btn-dot"></div>
      <span>${subtypeKey || name}</span>
      <span class="demo-btn-label">${label}</span>
    `;
    btn.addEventListener('click', () => runDemoSample(name));
    grid.appendChild(btn);
  });

  $('demoGrid').style.display = 'grid';
}

function guessSubtype(name) {
  const lower = name.toLowerCase();
  if (lower.includes('luma'))   return 'LumA';
  if (lower.includes('lumb'))   return 'LumB';
  if (lower.includes('her2'))   return 'Her2';
  if (lower.includes('basal'))  return 'Basal';
  if (lower.includes('normal')) return 'Normal';
  return null;
}

async function runDemoSample(name) {
  showLoading();
  $('resultsPanel').style.display = 'none';

  try {
    const resp = await fetch(`/demo/${name}`);
    const data = await resp.json();
    hideLoading();

    if (data.error) {
      flashError(data.error);
    } else {
      renderResults(data);
    }
  } catch (err) {
    hideLoading();
    flashError('Network error: ' + err.message);
  }
}

async function generateDemo() {
  const btn = $('generateBtn');
  btn.innerHTML = '<span>⏳ Generating…</span>';
  btn.disabled  = true;

  try {
    const resp = await fetch('/api/generate-demo');
    const data = await resp.json();

    if (data.error && !data.available) {
      flashError(data.error);
      btn.innerHTML = '<span>⚙ Generate from TCGA</span>';
      btn.disabled  = false;
      return;
    }

    if (data.available && data.available.length > 0) {
      $('demoEmpty').style.display = 'none';
      renderDemoGrid(data.available);
    } else if (data.error) {
      flashError(data.error);
      btn.innerHTML = '<span>⚙ Generate from TCGA</span>';
      btn.disabled  = false;
    }
  } catch (err) {
    flashError('Error: ' + err.message);
    btn.innerHTML = '<span>⚙ Generate from TCGA</span>';
    btn.disabled  = false;
  }
}

// ── Reset ────────────────────────────────────────────────────────────────────
function resetAll() {
  $('resultsPanel').style.display = 'none';
  dropZone.clear();
  document.getElementById('classify').scrollIntoView({ behavior: 'smooth' });
}

function clearFile() {
  dropZone.clear();
}

// ── Error flash ──────────────────────────────────────────────────────────────
function flashError(msg) {
  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:#ef4444; color:#fff; padding:12px 24px; border-radius:10px;
    font-size:.9rem; font-weight:500; z-index:9999;
    box-shadow:0 8px 30px rgba(239,68,68,0.4);
    animation: fadeInUp .3s ease;
  `;
  el.textContent = '⚠ ' + msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// ── Intersection Observer (reveal cards) ─────────────────────────────────────
function initRevealObserver() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        setTimeout(() => e.target.classList.add('visible'), i * 80);
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });

  document.querySelectorAll('.reveal-card').forEach(el => observer.observe(el));
}

// ── Smooth nav links ─────────────────────────────────────────────────────────
function initNavLinks() {
  document.querySelectorAll('.nav-link, .btn-hero').forEach(link => {
    link.addEventListener('click', e => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
}

// ── Chat Widget ─────────────────────────────────────────────────────────────
const chatWidget = (() => {
  let open    = false;
  let busy    = false;
  let history = [];

  function toggle() {
    open = !open;
    const panel = $('chatPanel');
    const bubble = $('chatBubble');
    if (open) {
      panel.style.display = 'flex';
      requestAnimationFrame(() => panel.classList.add('open'));
      $('chatUnread').style.display = 'none';
      bubble.classList.add('active');
      $('chatInput').focus();
    } else {
      panel.classList.remove('open');
      setTimeout(() => { panel.style.display = 'none'; }, 320);
      bubble.classList.remove('active');
    }
  }

  function appendMsg(text, role) {
    const wrap = document.createElement('div');
    wrap.className = `chat-msg ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble-msg';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    $('chatMessages').appendChild(wrap);
    $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
    return bubble;
  }

  function showTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'chat-msg bot';
    wrap.id = 'chatTyping';
    wrap.innerHTML = `<div class="chat-bubble-msg chat-typing">
      <span></span><span></span><span></span>
    </div>`;
    $('chatMessages').appendChild(wrap);
    $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
  }

  function hideTyping() {
    const el = $('chatTyping');
    if (el) el.remove();
  }

  async function send() {
    if (busy) return;
    const input = $('chatInput');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    appendMsg(message, 'user');

    busy = true;
    $('chatSendBtn').disabled = true;
    $('chatStatus').textContent = 'Thinking…';
    showTyping();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          history,
          context: window._lastPrediction || null,
        }),
      });
      const data = await resp.json();
      hideTyping();

      if (data.error) {
        appendMsg('⚠️ ' + data.error, 'bot');
      } else {
        appendMsg(data.reply, 'bot');
        history.push({ user: message, bot: data.reply });
      }
    } catch (err) {
      hideTyping();
      appendMsg('⚠️ Network error: ' + err.message, 'bot');
    } finally {
      busy = false;
      $('chatSendBtn').disabled = false;
      $('chatStatus').textContent = window._lastPrediction
        ? `Context: ${window._lastPrediction.subtype} patient`
        : 'Ask me anything about your results';
    }
  }

  return { toggle, send };
})();

// ── Init ─────────────────────────────────────────────────────────────────────
let dropZone;

document.addEventListener('DOMContentLoaded', () => {
  // Particle canvas
  new ParticleCanvas($('bgCanvas'));

  // Drop zone
  dropZone = new FileDropZone($('dropZone'), $('fileInput'));

  // Load demo list
  loadDemoList();

  // Reveal observer
  initRevealObserver();

  // Nav smooth scroll
  initNavLinks();
});

// Inject @keyframes for flash if needed
const style = document.createElement('style');
style.textContent = `
  @keyframes fadeInUp {
    from { opacity:0; transform:translateX(-50%) translateY(12px); }
    to   { opacity:1; transform:translateX(-50%) translateY(0); }
  }
`;
document.head.appendChild(style);