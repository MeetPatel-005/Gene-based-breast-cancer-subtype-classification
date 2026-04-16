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

  // 6. Sample preview table
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