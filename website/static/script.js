// ============================================
// script.js — Prediksi Tingkat Stres
// Frontend Logic: Sliders, Form, API, Results
// ============================================

(function () {
  'use strict';

  // ── Feature keys (same order as backend) ──
  const FEATURES = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
  ];

  // ── DOM refs ──
  const form = document.getElementById('stress-form');
  const btnSubmit = document.getElementById('btn-submit');
  const loadingOverlay = document.getElementById('loading-overlay');
  const hasilSection = document.getElementById('hasil');
  const stressCard = document.getElementById('stress-card');
  const stressIcon = document.getElementById('stress-icon');
  const stressLevel = document.getElementById('stress-level');
  const shapSection = document.getElementById('shap-section');
  const shapChart = document.getElementById('shap-chart');
  const saranSection = document.getElementById('saran-section');
  const saranList = document.getElementById('saran-list');
  const apresiasiSection = document.getElementById('apresiasi-section');
  const apresiasiTags = document.getElementById('apresiasi-tags');
  const btnNewAnalysis = document.getElementById('btn-new-analysis');

  // ══════════════════════════════════════════
  // 1. SLIDER VALUE DISPLAY
  // ══════════════════════════════════════════
  function initSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
      const valueDisplay = document.getElementById('val-' + slider.id);
      if (valueDisplay) {
        // Set initial
        valueDisplay.textContent = slider.value;

        // Update on input
        slider.addEventListener('input', () => {
          valueDisplay.textContent = slider.value;

          // Dynamic color based on value
          const val = parseInt(slider.value);
          if (val <= 3) {
            valueDisplay.style.color = '#10b981';
          } else if (val <= 6) {
            valueDisplay.style.color = '#06b6d4';
          } else {
            valueDisplay.style.color = '#f43f5e';
          }
        });

        // Trigger initial color
        slider.dispatchEvent(new Event('input'));
      }
    });
  }

  // ══════════════════════════════════════════
  // 2. TOGGLE BUTTONS (mental_health_history)
  // ══════════════════════════════════════════
  function initToggles() {
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    toggleBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const field = btn.dataset.field;
        const value = btn.dataset.value;

        // Deactivate siblings
        const siblings = document.querySelectorAll(`.toggle-btn[data-field="${field}"]`);
        siblings.forEach(s => s.classList.remove('active'));

        // Activate clicked
        btn.classList.add('active');

        // Update hidden input
        document.getElementById(field).value = value;
      });
    });
  }

  // ══════════════════════════════════════════
  // 3. SCROLL ANIMATIONS
  // ══════════════════════════════════════════
  function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
  }

  // ══════════════════════════════════════════
  // 4. FORM SUBMISSION
  // ══════════════════════════════════════════
  function initForm() {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      // Collect data
      const data = {};
      FEATURES.forEach(key => {
        const el = document.getElementById(key);
        data[key] = parseInt(el.value);
      });

      // Show loading
      loadingOverlay.classList.add('active');
      btnSubmit.disabled = true;

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(result.error || 'Terjadi kesalahan pada server.');
        }

        renderResults(result);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        loadingOverlay.classList.remove('active');
        btnSubmit.disabled = false;
      }
    });
  }

  // ══════════════════════════════════════════
  // 5. RENDER RESULTS
  // ══════════════════════════════════════════
  function renderResults(result) {
    // Show results section
    hasilSection.classList.add('visible');

    // ── Stress Level Card ──
    const prediksi = result.prediksi;

    // Clear old classes
    stressCard.className = 'stress-card';
    stressLevel.className = 'stress-level';

    if (prediksi === 'Tidak Stres') {
      stressCard.classList.add('tidak-stres');
      stressLevel.classList.add('tidak-stres');
      stressIcon.textContent = '😊';
    } else if (prediksi === 'Stres Ringan') {
      stressCard.classList.add('stres-ringan');
      stressLevel.classList.add('stres-ringan');
      stressIcon.textContent = '😟';
    } else {
      stressCard.classList.add('stres-berat');
      stressLevel.classList.add('stres-berat');
      stressIcon.textContent = '😰';
    }

    stressLevel.textContent = prediksi;

    // ── SHAP Chart ──
    if (result.shap_values && result.shap_values.length > 0) {
      shapSection.style.display = 'block';
      renderShapChart(result.shap_values);
    } else {
      shapSection.style.display = 'none';
    }

    // ── Saran ──
    saranList.innerHTML = '';
    if (result.saran && result.saran.length > 0) {
      result.saran.forEach((saran, i) => {
        const item = document.createElement('div');
        item.className = 'saran-item';

        // Parse category from "[Category] text" format
        const match = saran.match(/^\[(.+?)\]\s*(.+)$/);
        let html;
        if (match) {
          html = `
            <div class="saran-number">${i + 1}</div>
            <div class="saran-text">
              <span class="saran-category">[${match[1]}]</span> ${match[2]}
            </div>
          `;
        } else {
          html = `
            <div class="saran-number">${i + 1}</div>
            <div class="saran-text">${saran}</div>
          `;
        }

        item.innerHTML = html;
        saranList.appendChild(item);
      });
    }

    // ── Apresiasi ──
    if (result.apresiasi && result.apresiasi.length > 0) {
      apresiasiSection.style.display = 'block';
      apresiasiTags.innerHTML = '';
      result.apresiasi.forEach(text => {
        const tag = document.createElement('span');
        tag.className = 'apresiasi-tag';
        tag.textContent = '✓ ' + text.charAt(0).toUpperCase() + text.slice(1);
        apresiasiTags.appendChild(tag);
      });
    } else {
      apresiasiSection.style.display = 'none';
    }

    // Scroll to results
    setTimeout(() => {
      hasilSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }

  // ══════════════════════════════════════════
  // 6. SHAP BAR CHART
  // ══════════════════════════════════════════
  function renderShapChart(shapValues) {
    shapChart.innerHTML = '';

    // Find max absolute value for scaling
    const maxAbs = Math.max(...shapValues.map(v => Math.abs(v.nilai_shap)), 0.001);

    // Sort by absolute SHAP value descending
    const sorted = [...shapValues].sort((a, b) => Math.abs(b.nilai_shap) - Math.abs(a.nilai_shap));

    // Only show top 10 for readability
    const topItems = sorted.slice(0, 10);

    topItems.forEach((item, i) => {
      const row = document.createElement('div');
      row.className = 'shap-bar-row';

      const isPositive = item.nilai_shap >= 0;
      const widthPercent = (Math.abs(item.nilai_shap) / maxAbs) * 50;

      row.innerHTML = `
        <div class="shap-bar-label">${item.nama}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-center"></div>
          <div class="shap-bar-fill ${isPositive ? 'positive' : 'negative'}" 
               style="width: 0%"></div>
        </div>
        <div class="shap-bar-value ${isPositive ? 'positive' : 'negative'}">
          ${isPositive ? '+' : ''}${item.nilai_shap.toFixed(4)}
        </div>
      `;

      shapChart.appendChild(row);

      // Animate bar width
      requestAnimationFrame(() => {
        setTimeout(() => {
          const fill = row.querySelector('.shap-bar-fill');
          fill.style.width = widthPercent + '%';
        }, i * 60);
      });
    });
  }

  // ══════════════════════════════════════════
  // 7. NEW ANALYSIS BUTTON
  // ══════════════════════════════════════════
  function initNewAnalysis() {
    btnNewAnalysis.addEventListener('click', () => {
      // Hide results
      hasilSection.classList.remove('visible');

      // Reset form to defaults
      form.reset();

      // Re-init slider displays
      const sliders = document.querySelectorAll('input[type="range"]');
      sliders.forEach(slider => {
        slider.value = 5;
        slider.dispatchEvent(new Event('input'));
      });

      // Reset toggles
      const toggleDefaults = { mental_health_history: '0', blood_pressure: '2' };
      document.getElementById('mental_health_history').value = '0';
      document.getElementById('blood_pressure').value = '2';

      const toggleBtns = document.querySelectorAll('.toggle-btn');
      toggleBtns.forEach(btn => {
        const field = btn.dataset.field;
        const defaultVal = toggleDefaults[field];
        if (defaultVal !== undefined) {
          if (btn.dataset.value === defaultVal) {
            btn.classList.add('active');
          } else {
            btn.classList.remove('active');
          }
        }
      });

      // Scroll to form
      setTimeout(() => {
        document.getElementById('kuesioner').scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    });
  }

  // ══════════════════════════════════════════
  // 8. NAVBAR SCROLL EFFECT
  // ══════════════════════════════════════════
  function initNavbar() {
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
      const currentScroll = window.scrollY;

      if (currentScroll > 100) {
        navbar.style.background = 'rgba(10, 10, 26, 0.95)';
        navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
      } else {
        navbar.style.background = 'rgba(10, 10, 26, 0.8)';
        navbar.style.boxShadow = 'none';
      }

      lastScroll = currentScroll;
    });
  }

  // ══════════════════════════════════════════
  // INIT
  // ══════════════════════════════════════════
  document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    initToggles();
    initScrollAnimations();
    initForm();
    initNewAnalysis();
    initNavbar();
  });

})();
