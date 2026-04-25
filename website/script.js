/* =============================================================
   MambaRefine-CD Research Website — script.js
   ============================================================= */

/* ── Scroll progress bar ──────────────────────────────────── */
(function () {
  var bar = document.getElementById('progress-bar');
  if (!bar) return;
  window.addEventListener('scroll', function () {
    var scrolled = document.documentElement.scrollTop || document.body.scrollTop;
    var total    = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    bar.style.width = (total > 0 ? (scrolled / total) * 100 : 0) + '%';
  }, { passive: true });
}());

/* ── Mobile nav toggle ────────────────────────────────────── */
(function () {
  var toggle = document.querySelector('.nav-toggle');
  var links  = document.querySelector('.nav-links');
  if (!toggle || !links) return;
  toggle.addEventListener('click', function () {
    links.classList.toggle('open');
  });
  // Close on nav link click
  links.querySelectorAll('a').forEach(function (a) {
    a.addEventListener('click', function () {
      links.classList.remove('open');
    });
  });
}());

/* ── Smooth scroll with nav offset ───────────────────────── */
(function () {
  var navH = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--nav-h')) || 60;
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var id = a.getAttribute('href').slice(1);
      var el = document.getElementById(id);
      if (!el) return;
      e.preventDefault();
      var top = el.getBoundingClientRect().top + window.scrollY - navH - 12;
      window.scrollTo({ top: top, behavior: 'smooth' });
    });
  });
}());

/* ── Back-to-top button ───────────────────────────────────── */
(function () {
  var btn = document.getElementById('back-top');
  if (!btn) return;
  window.addEventListener('scroll', function () {
    btn.classList.toggle('visible', window.scrollY > 400);
  }, { passive: true });
  btn.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}());

/* ── Results tabs ─────────────────────────────────────────── */
(function () {
  document.querySelectorAll('.tabs-nav').forEach(function (nav) {
    var container = nav.closest('section') || nav.parentElement;
    nav.querySelectorAll('.tab-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = btn.dataset.tab;
        // deactivate all
        nav.querySelectorAll('.tab-btn').forEach(function (b) { b.classList.remove('active'); });
        container.querySelectorAll('.tab-panel').forEach(function (p) { p.classList.remove('active'); });
        // activate selected
        btn.classList.add('active');
        var panel = container.querySelector('#' + target);
        if (panel) panel.classList.add('active');
      });
    });
  });
}());

/* ── Math section expand/collapse ─────────────────────────── */
(function () {
  document.querySelectorAll('.math-toggle-btn').forEach(function (btn) {
    var contentId = btn.dataset.target;
    var content   = contentId ? document.getElementById(contentId) : btn.nextElementSibling;
    if (!content) return;
    btn.addEventListener('click', function () {
      var isOpen = content.classList.toggle('open');
      btn.textContent = isOpen ? '▲ Collapse full formulation' : '▼ Expand full formulation';
      if (isOpen && window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([content]);
      }
    });
  });
}());

/* ── Fade-in on scroll (IntersectionObserver) ─────────────── */
(function () {
  if (!window.IntersectionObserver) return;
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll('.fade-in').forEach(function (el) {
    obs.observe(el);
  });
}());
