// ---- Sync paired videos: when one finishes loading, rewind the
// other so the before/after pair stays in lockstep.
(function () {
  const vidloadsyncers = document.getElementsByClassName("vidloadsync");
  for (const elem of vidloadsyncers) {
    const srcshapeVid = elem.getElementsByClassName("srcshape").item(0);
    const rsltshapeVid = elem.getElementsByClassName("rsltshape").item(0);
    srcshapeVid?.addEventListener("loadeddata", e => {
      if (rsltshapeVid) {
        rsltshapeVid.currentTime = 0;
      }
    });
    rsltshapeVid?.addEventListener("loadeddata", e => {
      if (srcshapeVid) {
        srcshapeVid.currentTime = 0;
      }
    });
  }
})();

(function () {
  if (typeof BeforeAfter !== 'undefined') {
    new BeforeAfter({ id: '#example1' });
  }

  // ---- Heading anchor links (deep-link helpers) -----------------
  // For every <section id="..."> with an <h2>,
  // inject a small "#" link before the heading text so visitors
  // can hover to reveal and copy a permalink to that section.
  var sections = document.querySelectorAll('section[id] h2');
  sections.forEach(function (h) {
    var section = h.closest('section[id]');
    if (!section) return;
    if (h.querySelector('.heading-anchor')) return;
    var a = document.createElement('a');
    a.className = 'heading-anchor';
    a.href = '#' + section.id;
    a.setAttribute('aria-label', 'Permalink to ' + h.textContent.trim());
    a.textContent = '#';
    h.insertBefore(a, h.firstChild);
  });
})();

// ---- Side TOC + active section highlight ----------------------
// One IntersectionObserver tracks every <section id>; the section
// with the largest visible area "wins" and its corresponding
// entry in the side TOC gets .is-active.
(function () {
  var sections = Array.prototype.slice.call(document.querySelectorAll('section[id]'));
  var tocLinks = document.querySelectorAll('.side-toc a[data-section]');
  if (!sections.length) return;

  var visibility = Object.create(null);

  function setActive(id) {
    tocLinks.forEach(function (a) {
      a.classList.toggle('is-active', a.dataset.section === id);
    });
  }

  if ('IntersectionObserver' in window) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        visibility[entry.target.id] = entry.intersectionRatio;
      });
      var bestId = null;
      var bestRatio = 0;
      Object.keys(visibility).forEach(function (id) {
        if (visibility[id] > bestRatio) {
          bestRatio = visibility[id];
          bestId = id;
        }
      });
      if (bestId && bestRatio > 0) setActive(bestId);
    }, {
      // Treat the middle 60% of the viewport as the
      // "reading area"; whichever section overlaps it most
      // becomes the active one.
      rootMargin: '-20% 0px -20% 0px',
      threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
    });
    sections.forEach(function (s) { obs.observe(s); });
  }

  // Reveal the side TOC once the user has scrolled past the intro.
  var toc = document.querySelector('[data-toc]');
  if (toc) {
    var introH = (document.querySelector('.intro') || {}).offsetHeight || 400;
    var onScroll = function () {
      if (window.pageYOffset > introH * 0.6) toc.classList.add('is-visible');
      else toc.classList.remove('is-visible');
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }
})();

// ---- Image lightbox: click any content image to open a floating,
// zoomable copy. The white frame stays a fixed size (= initial fitted
// image size); zooming grows the image *inside* the frame, which clips
// it. At >1x the user can left-click-and-drag to pan. A click without
// drag closes the popup (as does any click on the dim backdrop).
(function () {
  var lightbox = document.getElementById('image-lightbox');
  if (!lightbox) return;
  var stage = lightbox.querySelector('[data-stage]');
  var frameEl = lightbox.querySelector('[data-frame]');
  var contentEl = lightbox.querySelector('[data-content]');
  var imgEl = lightbox.querySelector('.lightbox-img');
  var captionEl = lightbox.querySelector('[data-caption-mount]');
  var modelFrameEl = lightbox.querySelector('[data-model-frame]');
  var modelEl = lightbox.querySelector('[data-model]');
  var closeBtn = lightbox.querySelector('.lightbox-close');
  var controls = lightbox.querySelectorAll('.lightbox-btn');

  var MIN_SCALE = 1;
  var MAX_SCALE = 8;
  var DRAG_THRESHOLD = 5;
  var BASE_CAPTION_FONT_REM = 1.1;     // matches .caption default
  // Caption block = .caption-cols height (1.4em) plus .caption margin-top (0.4rem).
  var CAPTION_BLOCK_EM = 1.4;
  var CAPTION_MARGIN_REM = 0.4;
  var scale = 1;
  var baseW = 0, baseH = 0;            // image's fitted size at scale 1
  var baseCaptionH = 0;                // caption block height at scale 1
  var hasCaption = false;
  var offsetX = 0, offsetY = 0;        // pan offset within frame
  var dragging = false, didDrag = false;
  var pointerStart = null, offsetXStart = 0, offsetYStart = 0;
  var lastFocused = null;

  function getRem() {
    return parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  }

  function captionHeightAt(scaleFactor) {
    if (!hasCaption) return 0;
    var rem = getRem();
    return (CAPTION_BLOCK_EM * BASE_CAPTION_FONT_REM + CAPTION_MARGIN_REM) * rem * scaleFactor;
  }

  function computeFit() {
    var rem = getRem();
    var borderPx = 0.6 * rem;
    var maxW = window.innerWidth * 0.92 - 2 * borderPx;
    var maxH = window.innerHeight * 0.88 - 2 * borderPx;
    baseCaptionH = captionHeightAt(1);
    // Reserve room for the caption so the image still fits.
    var imgMaxH = Math.max(40, maxH - baseCaptionH);
    var nW = imgEl.naturalWidth;
    var nH = imgEl.naturalHeight;
    if (!nW || !nH) { baseW = maxW; baseH = imgMaxH; return; }
    var ratio = Math.min(maxW / nW, imgMaxH / nH, 1);
    baseW = nW * ratio;
    baseH = nH * ratio;
  }

  function canPan() {
    var contentW = baseW * scale;
    var contentH = baseH * scale + captionHeightAt(scale);
    var fW = parseFloat(frameEl.style.width) || baseW;
    var fH = parseFloat(frameEl.style.height) || baseH;
    return contentW > fW + 0.5 || contentH > fH + 0.5;
  }

  function updateCursor() {
    if (canPan()) frameEl.style.cursor = dragging ? 'grabbing' : 'grab';
    else frameEl.style.cursor = 'zoom-out';
  }

  function apply() {
    if (!(baseW > 0 && baseH > 0)) return;
    var rem = getRem();
    var borderPx = 0.6 * rem;
    var maxH = window.innerHeight * 0.88 - 2 * borderPx;
    // Width is locked at the fitted baseline; height grows with zoom
    // (up to the viewport cap) so wider figures reveal more vertical
    // content before you have to drag-to-pan.
    var imgW = baseW * scale;
    var imgH = baseH * scale;
    var capH = captionHeightAt(scale);
    var contentH = imgH + capH;
    var frameW = baseW;
    var frameH = Math.min(contentH, maxH);
    frameEl.style.width = frameW + 'px';
    frameEl.style.height = frameH + 'px';
    // Setting the content width drives the image (width: 100%) and any
    // caption inside. SVGs re-rasterize crisply because the browser
    // re-renders them at the new layout size.
    contentEl.style.width = imgW + 'px';
    if (hasCaption) {
      captionEl.style.fontSize = (BASE_CAPTION_FONT_REM * scale) + 'rem';
    }
    // Clamp pan so the content always covers the frame.
    var maxOX = Math.max(0, (imgW - frameW) / 2);
    var maxOY = Math.max(0, (contentH - frameH) / 2);
    offsetX = Math.max(-maxOX, Math.min(maxOX, offsetX));
    offsetY = Math.max(-maxOY, Math.min(maxOY, offsetY));
    contentEl.style.transform =
      'translate(-50%, -50%) translate(' + offsetX + 'px, ' + offsetY + 'px)';
    updateCursor();
  }

  function reset() {
    scale = 1; offsetX = 0; offsetY = 0;
    apply();
  }

  // originX/originY (viewport coords) anchor the zoom so the image point
  // under the cursor stays fixed. Omitted -> zoom around the frame center.
  function zoomBy(factor, originX, originY) {
    var next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
    if (next === scale) return;
    var ratio = next / scale;
    if (originX != null && originY != null && next > 1) {
      var rect = frameEl.getBoundingClientRect();
      var dx = originX - (rect.left + rect.width / 2);
      var dy = originY - (rect.top + rect.height / 2);
      offsetX = dx - (dx - offsetX) * ratio;
      offsetY = dy - (dy - offsetY) * ratio;
    }
    scale = next;
    if (scale === 1) { offsetX = 0; offsetY = 0; }
    apply();
  }

  // When suppressFitReset is set, the next imgEl load is a streamlines
  // hover/leave swap — keep the current zoom/pan. computeFit still runs
  // so the frame stays correctly sized to the new image's aspect.
  var suppressFitReset = false;
  imgEl.addEventListener('load', function () {
    computeFit();
    if (suppressFitReset) {
      suppressFitReset = false;
      apply();
    } else {
      reset();
    }
  });

  function open(src, alt, captionHTML, captionAbove) {
    lastFocused = document.activeElement;
    scale = 1; offsetX = 0; offsetY = 0;
    baseW = 0; baseH = 0; baseCaptionH = 0;
    suppressFitReset = false;
    contentEl.style.width = '';
    contentEl.style.transform = '';
    captionEl.innerHTML = captionHTML || '';
    captionEl.style.fontSize = '';
    hasCaption = !!captionHTML;
    // Place the cloned caption above or below the image to match the
    // source figure's layout.
    if (captionAbove) contentEl.insertBefore(captionEl, imgEl);
    else contentEl.appendChild(captionEl);
    imgEl.src = src;
    imgEl.alt = alt || '';
    lightbox.classList.add('is-open');
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    closeBtn.focus();
  }

  function close() {
    lightbox.classList.remove('is-open');
    lightbox.classList.remove('is-model-mode');
    lightbox.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    clearStreamlinesSwap();
    imgEl.src = '';
    modelEl.removeAttribute('src');
    captionEl.innerHTML = '';
    hasCaption = false;
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  // ---- Streamlines pair swap: when a streamlines popup is open, the
  // frame's mouseenter/leave swap between base and streamlines so the
  // popup mirrors the on-page hover behavior.
  var streamlinesSwap = null;
  function clearStreamlinesSwap() {
    if (!streamlinesSwap) return;
    frameEl.removeEventListener('mouseenter', streamlinesSwap.enter);
    frameEl.removeEventListener('mouseleave', streamlinesSwap.leave);
    streamlinesSwap = null;
  }

  // ---- Open in model-viewer (3D) mode. Clones key attributes from the
  // page model-viewer so the popup shows the same model with the same
  // camera-orbit / shadow / exposure defaults.
  function openModel(source) {
    lastFocused = document.activeElement;
    var attrs = ['src', 'alt', 'camera-orbit', 'shadow-intensity',
                 'exposure', 'rotation-per-second'];
    attrs.forEach(function (name) {
      if (source.hasAttribute(name)) {
        modelEl.setAttribute(name, source.getAttribute(name));
      } else {
        modelEl.removeAttribute(name);
      }
    });
    lightbox.classList.add('is-open');
    lightbox.classList.add('is-model-mode');
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    closeBtn.focus();
  }

  // ---- Wire up triggers: every content <img> opens the lightbox.
  // If the image lives inside a <figure> that has column-label captions
  // (.caption.caption-cols), clone those into the popup so the labels
  // stay attached to the image as it zooms.
  var triggers = document.querySelectorAll(
    'img:not(.lightbox-img):not(.side-toc img):not([data-no-zoom])'
  );
  triggers.forEach(function (img) {
    img.classList.add('is-zoomable');
    img.addEventListener('click', function () {
      var fig = img.closest('figure');
      var cap = fig && fig.querySelector('.caption.caption-cols');
      // True if the caption sits before the image in the source figure.
      var capAbove = !!(cap && (cap.compareDocumentPosition(img) & Node.DOCUMENT_POSITION_FOLLOWING));
      open(img.currentSrc || img.src, img.alt, cap ? cap.outerHTML : null, capAbove);
    });
  });

  // ---- Streamlines pair: clicking opens the streamlines image; the
  // popup frame's mouseleave reverts to the base image, mouseenter swaps
  // back to streamlines. Marked data-no-zoom so the generic single-image
  // trigger above doesn't double-fire.
  document.querySelectorAll('.streamlines-img-wrap').forEach(function (wrap) {
    var base = wrap.querySelector('.streamlines-base');
    var hover = wrap.querySelector('.streamlines-hover');
    if (!base || !hover) return;
    wrap.style.cursor = 'zoom-in';
    wrap.addEventListener('click', function () {
      open(base.src, base.alt || '', null, false);
      var enter = function () { suppressFitReset = true; imgEl.src = hover.src; };
      var leave = function () { suppressFitReset = true; imgEl.src = base.src; };
      clearStreamlinesSwap();
      frameEl.addEventListener('mouseenter', enter);
      frameEl.addEventListener('mouseleave', leave);
      streamlinesSwap = { enter: enter, leave: leave };
    });
  });

  // ---- [data-zoom-pair] wrapper: clicking anywhere inside opens one
  // lightbox showing both child images stitched side-by-side as a single
  // composite SVG so they zoom as one. Implementation: fetch each source
  // SVG, prefix all internal ids (so url(#x) refs from different files
  // don't collide), then drop each into a nested <svg> with its own
  // viewBox. Everything stays vector, so zoom remains crisp. Result is
  // cached per wrapper as a blob URL.
  var pairCache = new WeakMap();
  function prefixIdsInSvg(svgRoot, prefix) {
    var idMap = Object.create(null);
    svgRoot.querySelectorAll('[id]').forEach(function (el) {
      var oldId = el.getAttribute('id');
      var newId = prefix + oldId;
      idMap[oldId] = newId;
      el.setAttribute('id', newId);
    });
    var urlRefRe = /url\(\s*#([^)\s]+)\s*\)/g;
    var refAttrs = ['fill', 'stroke', 'clip-path', 'mask', 'filter', 'style', 'marker-start', 'marker-mid', 'marker-end'];
    svgRoot.querySelectorAll('*').forEach(function (el) {
      refAttrs.forEach(function (a) {
        var v = el.getAttribute(a);
        if (v == null || v.indexOf('url(') < 0) return;
        var nv = v.replace(urlRefRe, function (m, id) {
          return idMap[id] ? 'url(#' + idMap[id] + ')' : m;
        });
        if (nv !== v) el.setAttribute(a, nv);
      });
      ['href', 'xlink:href'].forEach(function (a) {
        var v = el.getAttribute(a);
        if (v && v.charAt(0) === '#') {
          var id = v.slice(1);
          if (idMap[id]) el.setAttribute(a, '#' + idMap[id]);
        }
      });
    });
  }
  async function buildPairSrc(wrapper) {
    if (pairCache.has(wrapper)) return pairCache.get(wrapper);
    var imgs = Array.prototype.slice.call(wrapper.querySelectorAll('img'));
    var texts = await Promise.all(imgs.map(function (i) {
      return fetch(i.src).then(function (r) { return r.text(); });
    }));
    var parser = new DOMParser();
    var roots = texts.map(function (t, i) {
      var root = parser.parseFromString(t, 'image/svg+xml').documentElement;
      prefixIdsInSvg(root, 's' + i + '_');
      return root;
    });
    var dims = roots.map(function (svg) {
      var vb = svg.getAttribute('viewBox');
      var w, h;
      if (vb) {
        var p = vb.trim().split(/[\s,]+/).map(Number);
        w = p[2]; h = p[3];
      } else {
        w = parseFloat(svg.getAttribute('width')) || 100;
        h = parseFloat(svg.getAttribute('height')) || 100;
      }
      return { w: w, h: h, viewBox: vb };
    });
    var H = 1000;
    var widths = dims.map(function (d) { return d.w * (H / d.h); });
    var gap = H * 0.015;
    var totalW = widths.reduce(function (a, b) { return a + b; }, 0)
               + gap * (roots.length - 1);
    var ns = 'http://www.w3.org/2000/svg';
    var combined = document.createElementNS(ns, 'svg');
    combined.setAttribute('xmlns', ns);
    combined.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    combined.setAttribute('viewBox', '0 0 ' + totalW + ' ' + H);
    combined.setAttribute('width', totalW);
    combined.setAttribute('height', H);
    var x = 0;
    roots.forEach(function (root, i) {
      var wrap = document.createElementNS(ns, 'svg');
      wrap.setAttribute('x', x);
      wrap.setAttribute('y', 0);
      wrap.setAttribute('width', widths[i]);
      wrap.setAttribute('height', H);
      if (dims[i].viewBox) wrap.setAttribute('viewBox', dims[i].viewBox);
      wrap.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      while (root.firstChild) wrap.appendChild(root.firstChild);
      combined.appendChild(wrap);
      x += widths[i] + gap;
    });
    var xml = new XMLSerializer().serializeToString(combined);
    var url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml' }));
    pairCache.set(wrapper, url);
    return url;
  }
  document.querySelectorAll('[data-zoom-pair]').forEach(function (wrapper) {
    wrapper.addEventListener('click', function () {
      buildPairSrc(wrapper).then(function (src) {
        open(src, wrapper.getAttribute('aria-label') || '', null, false);
      }).catch(function (err) {
        console.error('Failed to build combined image:', err);
      });
    });
  });

  // ---- Single left-click on a content <model-viewer> opens the popup.
  // We distinguish a click from a drag (used by model-viewer for orbit
  // rotation) by tracking pointer movement: if the pointer barely moved
  // between down and up, it's a click; otherwise it's a drag and we
  // leave it to model-viewer's own controls.
  var modelTriggers = document.querySelectorAll('model-viewer:not([data-model])');
  var MV_CLICK_THRESHOLD = 5;
  modelTriggers.forEach(function (mv) {
    var startX = 0, startY = 0, tracking = false, moved = false;
    mv.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;       // left button only
      tracking = true;
      moved = false;
      startX = e.clientX;
      startY = e.clientY;
    });
    mv.addEventListener('pointermove', function (e) {
      if (!tracking || moved) return;
      if (Math.hypot(e.clientX - startX, e.clientY - startY) > MV_CLICK_THRESHOLD) {
        moved = true;
      }
    });
    mv.addEventListener('pointerup', function (e) {
      if (e.button !== 0 || !tracking) return;
      var wasClick = !moved;
      tracking = false;
      if (wasClick) openModel(mv);
    });
    mv.addEventListener('pointercancel', function () { tracking = false; });
    mv.style.cursor = 'zoom-in';
  });

  // Clicks / wheel / pointer events inside the lightbox model-viewer
  // belong to the 3D controls, not to the lightbox — stop them from
  // bubbling up to the close-on-click handler.
  ['click', 'pointerdown', 'wheel'].forEach(function (evt) {
    modelFrameEl.addEventListener(evt, function (e) { e.stopPropagation(); });
  });

  // ---- Close interactions. Backdrop / frame clicks bubble up to here.
  lightbox.addEventListener('click', close);
  closeBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    close();
  });
  controls.forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      var action = btn.dataset.action;
      if (action === 'zoom-in') zoomBy(1.4);
      else if (action === 'zoom-out') zoomBy(1 / 1.4);
      else if (action === 'zoom-reset') reset();
    });
  });

  document.addEventListener('keydown', function (e) {
    if (!lightbox.classList.contains('is-open')) return;
    if (e.key === 'Escape') { close(); }
    else if (e.key === '+' || e.key === '=') { zoomBy(1.25); }
    else if (e.key === '-' || e.key === '_') { zoomBy(1 / 1.25); }
    else if (e.key === '0') { reset(); }
  });

  // ---- Wheel zooms in place (centered). Frame size doesn't change.
  // In model-viewer mode the 3D viewer handles its own scroll-to-zoom,
  // so we skip the image-zoom logic entirely.
  stage.addEventListener('wheel', function (e) {
    if (!lightbox.classList.contains('is-open')) return;
    if (lightbox.classList.contains('is-model-mode')) return;
    e.preventDefault();
    e.stopPropagation();
    zoomBy(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY);
  }, { passive: false });

  // ---- Drag-to-pan on the frame, only when zoomed past 1x. A click
  // without movement bubbles up and closes the popup as usual.
  frameEl.addEventListener('pointerdown', function (e) {
    if (e.button !== 0) return;
    pointerStart = { x: e.clientX, y: e.clientY };
    didDrag = false;
    if (canPan()) {
      dragging = true;
      offsetXStart = offsetX;
      offsetYStart = offsetY;
      try { frameEl.setPointerCapture(e.pointerId); } catch (_) {}
      updateCursor();
    }
  });

  frameEl.addEventListener('pointermove', function (e) {
    if (!pointerStart) return;
    var dx = e.clientX - pointerStart.x;
    var dy = e.clientY - pointerStart.y;
    if (!didDrag && Math.hypot(dx, dy) > DRAG_THRESHOLD) didDrag = true;
    if (dragging) {
      offsetX = offsetXStart + dx;
      offsetY = offsetYStart + dy;
      apply();
    }
  });

  function endDrag(e) {
    if (!pointerStart && !dragging) return;
    var wasDragging = dragging;
    dragging = false;
    pointerStart = null;
    try {
      if (frameEl.hasPointerCapture && frameEl.hasPointerCapture(e.pointerId)) {
        frameEl.releasePointerCapture(e.pointerId);
      }
    } catch (_) {}
    if (wasDragging) apply(); else updateCursor();
  }
  frameEl.addEventListener('pointerup', endDrag);
  frameEl.addEventListener('pointercancel', endDrag);

  // After a real drag, swallow the trailing click so it doesn't close.
  frameEl.addEventListener('click', function (e) {
    if (didDrag) {
      e.stopPropagation();
      didDrag = false;
    }
  });
})();

// ---- Auto-reset model-viewer camera once it scrolls offscreen.
// When a viewer leaves the viewport, reset cameraOrbit/Target/fieldOfView
// so the next time it scrolls back into view it's framed at its starting
// pose, in sync with the built-in auto-rotate restart.
(function () {
  if (!('IntersectionObserver' in window)) return;
  var viewers = document.querySelectorAll('model-viewer[auto-rotate]');
  if (!viewers.length) return;
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) return;
      var mv = entry.target;
      mv.cameraOrbit = 'auto auto auto';
      mv.cameraTarget = 'auto auto auto';
      mv.fieldOfView = 'auto';
    });
  });
  viewers.forEach(function (mv) { obs.observe(mv); });
})();

// ---- BibTeX copy-to-clipboard ---------------------------------
(function () {
  var btn = document.querySelector('.bibtex-copy-btn');
  if (!btn) return;
  var resetTimer = null;
  btn.addEventListener('click', function () {
    var target = document.getElementById(btn.dataset.target);
    if (!target) return;
    var text = target.innerText;
    var done = function (ok) {
      btn.classList.remove('is-success', 'is-error');
      btn.classList.add(ok ? 'is-success' : 'is-error');
      var label = btn.querySelector('.bibtex-copy-text');
      if (label) label.textContent = ok ? 'Copied!' : 'Copy failed';
      if (resetTimer) clearTimeout(resetTimer);
      resetTimer = setTimeout(function () {
        btn.classList.remove('is-success', 'is-error');
        if (label) label.textContent = 'Copy';
      }, 1800);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); },
        function () { done(false); });
    } else {
      try {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        var ok = document.execCommand('copy');
        document.body.removeChild(ta);
        done(ok);
      } catch (err) {
        done(false);
      }
    }
  });
})();
