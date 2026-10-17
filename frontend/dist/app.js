(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const demoMode = new URLSearchParams(location.search).get('demo') === '1';
  const client = window.ForteLensAnalysis;
  const state = { source: null, file: null, phase: 'idle', result: null, error: '', controller: null, revision: 0, exporting: false, exportController: null };
  const icons = {
    upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>',
    film: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 3v18M17 3v18M3 8h4m-4 8h4m10-8h4m-4 8h4"/>',
    x: '<path d="m6 6 12 12M6 18 18 6"/>',
    shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8 12 3 3 5-6"/>',
    scan: '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3M7 12h10M12 7v10"/>',
    'arrow-right': '<path d="M4 12h16m-6-6 6 6-6 6"/>',
    'arrow-left': '<path d="M20 12H4m6-6-6 6 6 6"/>',
    download: '<path d="M12 3v13m-5-5 5 5 5-5M4 17v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 13h8m-8 4h6"/>',
    book: '<path d="M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5"/>',
    external: '<path d="M14 3h7v7m0-7L11 13M9 5H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
    alert: '<path d="m10.3 4.8-8 14A1.5 1.5 0 0 0 3.6 21h16.8a1.5 1.5 0 0 0 1.3-2.2l-8-14a2 2 0 0 0-3.4 0Z"/><path d="M12 10v5m0 3v.1"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="m7 12 3 3 7-7"/>'
  };
  const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.info}</svg>`;
  function renderIcons(root = document) { root.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon)); }
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function timecode(value) {
    const ticks = Math.round(value * 10), hours = Math.floor(ticks / 36000), minutes = Math.floor(ticks / 600) % 60, seconds = Math.floor(ticks / 10) % 60;
    return `${hours ? String(hours).padStart(2, '0') + ':' : ''}${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${ticks % 10}`;
  }
  const plural = n => n % 10 === 1 && n % 100 !== 11 ? 'нарушение' : n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14) ? 'нарушения' : 'нарушений';
  const hasReport = () => state.phase === 'done' && state.result?.verdict === 'violation' && state.result.report.violations.length > 0;
  let toastTimer;
  function toast(message) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false; toastTimer = setTimeout(() => $('toast').hidden = true, 4200); }
  function clearResult() {
    state.result = null;
    state.error = '';
    $('violation-list').replaceChildren();
    $('report-file').textContent = '';
    $('report-date').textContent = '';
    $('report-count').textContent = '';
    $('pdf-error').hidden = true;
    if ($('frame-dialog').open) $('frame-dialog').close();
    $('frame-image').removeAttribute('src');
  }
  function invalidate() {
    state.revision++;
    state.controller?.abort();
    state.controller = null;
    state.exportController?.abort();
    state.exportController = null;
    state.exporting = false;
    [$('download-pdf'), $('download-pdf-bottom')].forEach(button => { button.disabled = false; button.querySelector('span').textContent = 'Скачать PDF'; });
    clearResult();
  }
  function resetSelection() {
    invalidate();
    state.source = null;
    state.file = null;
    state.phase = 'idle';
    $('video-input').value = '';
    $('file-error').hidden = true;
    renderUpload();
    route(false);
  }
  function renderUpload() {
    const busy = state.phase === 'processing';
    const selected = Boolean(state.source);
    const ready = hasReport();
    $('selected-file').hidden = !selected;
    $('analysis-actions').hidden = !selected || busy || state.phase === 'done';
    $('processing').hidden = !busy;
    $('dropzone').disabled = busy;
    $('video-input').disabled = busy;
    $('remove-file').disabled = busy;
    $('demo-scenario').disabled = busy;
    $('load-demo').disabled = busy;
    $('analyze-button').disabled = !selected || busy || (state.source === 'file' && !client.available);
    $('analyze-button').querySelector('span').textContent = state.phase === 'error' ? 'Повторить проверку' : state.source === 'demo' ? 'Проверить учебный пример' : 'Проверить видео';
    $('analysis-hint').textContent = state.source === 'demo' ? 'Демонстрация переходов. Модель не запускается.' : client.available ? 'Результат появится после завершения анализа.' : 'Видео выбрано. Проверка станет доступна после подключения ИИ.';
    $('analysis-result').hidden = !['done', 'error'].includes(state.phase);
    // Hidden and disabled are intentional: both UI and routing use the same guard.
    $('view-violations').hidden = !ready;
    $('view-violations').disabled = !ready;
    if (selected) {
      $('file-name').textContent = state.source === 'demo' ? window.ForteLensDemo.fileName : state.file.name;
      $('file-meta').textContent = state.source === 'demo' ? 'Учебные схемы · не видеозапись' : state.file.size < 1024 * 1024 ? `${Math.max(1, Math.ceil(state.file.size / 1024))} КБ · ${state.file.name.split('.').pop().toUpperCase()}` : `${(state.file.size / 1024 / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} МБ · ${state.file.name.split('.').pop().toUpperCase()}`;
    }
    if (state.phase === 'error') {
      $('analysis-result').dataset.tone = 'error';
      $('result-icon').innerHTML = icon('alert');
      $('result-heading').textContent = 'Не удалось проверить видео';
      $('result-message').textContent = state.error;
    } else if (state.phase === 'done') {
      $('analysis-result').dataset.tone = ready ? 'violation' : 'clear';
      $('result-icon').innerHTML = icon(ready ? 'alert' : 'check');
      const prefix = state.source === 'demo' ? 'Демо: ' : '';
      const count = state.result.report.violations.length;
      $('result-heading').textContent = ready ? `${prefix}найдено ${count} ${plural(count)}` : `${prefix}нарушений не обнаружено`;
      $('result-message').textContent = ready ? 'Откройте отчёт: кадры, время, описание и нормативные основания.' : 'Проверка завершена без обнаруженных событий. Отчёт о нарушениях недоступен.';
    }
  }
  function selectFile(file) {
    if (!file || state.phase === 'processing') return;
    // Any newly selected file invalidates an earlier report, including invalid input.
    resetSelection();
    let error = '';
    if (!/\.(mov|mp4|webm)$/i.test(file.name)) error = 'Выберите видео MOV, MP4 или WebM.';
    else if (file.size === 0) error = 'Файл пуст. Выберите другую видеозапись.';
    else if (file.size > 100 * 1024 * 1024) error = 'Размер файла больше 100 МБ. Выберите запись меньшего размера.';
    if (error) { $('file-error').textContent = error; $('file-error').hidden = false; return; }
    state.source = 'file'; state.file = file; state.phase = 'ready';
    renderUpload();
    if (client.available) $('analyze-button').focus();
  }
  function selectDemo() {
    if (!demoMode || state.phase === 'processing') return;
    resetSelection();
    state.source = 'demo'; state.phase = 'ready';
    renderUpload(); $('analyze-button').focus();
  }
  function safeLink(value) {
    const url = new URL(value);
    if (!['https:', 'http:'].includes(url.protocol)) throw new Error('Некорректная ссылка на нормативный источник.');
    return url.href;
  }
  function safeFrame(value) {
    if (typeof value !== 'string') throw new Error('Некорректный кадр в результате анализа.');
    if (/^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=\r\n]+$/.test(value)) return value;
    const url = new URL(value, location.href);
    if (url.origin === location.origin && /^https?:$/.test(url.protocol) && url.pathname.includes('/assets/')) return url.href;
    if (location.protocol === 'file:' && /^assets\/[a-zA-Z0-9_.\/-]+$/.test(value) && !value.includes('..')) return value;
    throw new Error('Кадры должны быть переданы изображениями или локальными ресурсами сайта.');
  }
  function normalizeResult(raw, source, name) {
    if (!raw || !['violation', 'no_violation'].includes(raw.verdict) || !Array.isArray(raw.violations)) throw new Error('Сервис вернул неполный результат. Повторите проверку.');
    if ((raw.verdict === 'violation') !== (raw.violations.length > 0)) throw new Error('Результат и список нарушений не совпадают. Повторите проверку.');
    const text = (value, field) => { if (typeof value !== 'string' || !value.trim()) throw new Error(`В результате отсутствует ${field}.`); return value.trim(); };
    const violations = raw.violations.map((v, index) => {
      if (!v || !Number.isFinite(v.timecode_sec) || v.timecode_sec < 0 || !Array.isArray(v.frames) || v.frames.length < 1 || v.frames.length > 3) throw new Error('В результате отсутствуют корректные кадры или время события.');
      return {
        id: String(v.id || `event-${index + 1}`),
        title: text(v.title, 'название нарушения'), timecode_sec: v.timecode_sec,
        description: text(v.description, 'описание нарушения'),
        frames: v.frames.map((frame, frameIndex) => ({ src: safeFrame(frame?.src), caption: typeof frame.caption === 'string' && frame.caption.trim() ? frame.caption : `Кадр ${frameIndex + 1}` })),
        legal: { act: text(v.legal?.act, 'нормативный акт'), paragraph: text(v.legal?.paragraph, 'точный пункт'), url: safeLink(text(v.legal?.url, 'ссылка на источник')), context: typeof v.legal?.context === 'string' ? v.legal.context : '' }
      };
    });
    return { verdict: raw.verdict, report: { fileName: name, createdAt: new Date().toISOString(), isDemo: source === 'demo', violations } };
  }
  async function analyze() {
    if (!state.source || state.phase === 'processing' || (state.source === 'file' && !client.available)) return;
    invalidate();
    const revision = state.revision, source = state.source;
    const name = source === 'demo' ? window.ForteLensDemo.fileName : state.file.name;
    const controller = new AbortController(); state.controller = controller;
    state.phase = 'processing';
    $('processing-title').textContent = source === 'demo' ? 'Проверяем учебный сценарий' : 'Проверяем видео';
    $('processing-message').textContent = 'Ожидаем результат…';
    renderUpload(); $('cancel-button').focus({ preventScroll: true });
    try {
      const options = { signal: controller.signal, onStatus: message => { if (state.revision === revision && !controller.signal.aborted) $('processing-message').textContent = String(message); } };
      const raw = source === 'demo' ? await window.ForteLensDemo.analyze($('demo-scenario').value, options) : await client.analyze(state.file, options);
      if (revision !== state.revision || controller.signal.aborted) return;
      state.result = normalizeResult(raw, source, name);
      state.phase = 'done';
    } catch (error) {
      if (revision !== state.revision || controller.signal.aborted) return;
      state.phase = 'error'; state.result = null;
      state.error = error instanceof Error ? error.message : 'Не удалось получить результат. Повторите проверку.';
    } finally {
      if (revision === state.revision) {
        state.controller = null; renderUpload();
        const target = hasReport() ? $('view-violations') : $('result-heading');
        target.setAttribute('tabindex', hasReport() ? '0' : '-1'); target.focus({ preventScroll: true });
      }
    }
  }
  function renderReport() {
    if (!hasReport()) return;
    const report = state.result.report;
    $('report-heading').textContent = report.isDemo ? 'Нарушения · учебный пример' : 'Обнаруженные нарушения';
    $('report-file').textContent = report.fileName;
    $('report-date').dateTime = report.createdAt;
    $('report-date').textContent = new Date(report.createdAt).toLocaleString('ru-RU', { day: '2-digit', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    $('report-count').textContent = `${report.violations.length} ${plural(report.violations.length)}`;
    $('report-demo-note').hidden = !report.isDemo;
    $('violation-list').innerHTML = report.violations.map((v, i) => `<article class="violation-card" aria-labelledby="violation-${i}"><header class="violation-header"><h2 id="violation-${i}">Нарушение ${String(i + 1).padStart(2, '0')}</h2><span class="event-status">${report.isDemo ? 'Учебный пример' : 'Обнаружено при анализе'}</span></header><div class="frames">${v.frames.map((frame, f) => `<figure class="frame"><button type="button" class="frame-button" data-event="${i}" data-frame="${f}" aria-label="Увеличить кадр ${f + 1} нарушения ${i + 1}"><img src="${escape(frame.src)}" alt="${escape(frame.caption)}"><span class="frame-zoom">${icon('expand')}</span></button><figcaption>${escape(frame.caption)}</figcaption></figure>`).join('')}</div><div class="violation-body"><div class="time-label">${icon('clock')}Время на видео<strong>${timecode(v.timecode_sec)}</strong></div><div class="description"><h3>${escape(v.title)}</h3><p>${escape(v.description)}</p></div><section class="legal" aria-label="Нормативное основание"><h3>${icon('book')}Основание</h3><dl><dt>Акт</dt><dd>${escape(v.legal.act)}</dd><dt>Точный пункт</dt><dd><strong>${escape(v.legal.paragraph)}</strong></dd><dt>Источник</dt><dd><a class="source-link" href="${escape(v.legal.url)}" target="_blank" rel="noopener noreferrer">Открыть нормативный акт${icon('external')}</a></dd></dl>${v.legal.context ? `<p class="legal-context">${escape(v.legal.context)}</p>` : ''}</section></div></article>`).join('');
    $('violation-list').querySelectorAll('img').forEach(img => img.addEventListener('error', () => { img.alt = 'Кадр не загрузился. PDF недоступен до восстановления изображения.'; }));
  }
  function route(focus = true) {
    const requested = location.hash === '#report';
    const showReport = requested && hasReport();
    if (requested && !showReport) {
      history.replaceState(null, '', location.pathname + location.search + '#upload');
      toast('Сначала загрузите видео и дождитесь результата с найденным нарушением.');
    }
    $('upload-view').hidden = showReport;
    $('report-view').hidden = !showReport;
    document.title = `ForteLens · ${showReport ? 'Отчёт о нарушениях' : 'Загрузка видео'}`;
    if (showReport) renderReport();
    if (focus) $(showReport ? 'report-heading' : 'upload-heading').focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: 'instant' });
  }
  async function downloadPDF() {
    if (!hasReport() || state.exporting) return;
    const revision = state.revision;
    const exportController = new AbortController();
    state.exportController = exportController;
    state.exporting = true;
    $('pdf-error').hidden = true;
    const buttons = [$('download-pdf'), $('download-pdf-bottom')];
    buttons.forEach(button => { button.disabled = true; button.querySelector('span').textContent = 'Готовим PDF…'; });
    try {
      if (!window.ForteLensPDF) throw new Error('Не удалось загрузить модуль PDF. Обновите страницу и попробуйте ещё раз.');
      await window.ForteLensPDF.download(state.result.report, { signal: exportController.signal });
      if (revision === state.revision && !exportController.signal.aborted) toast('PDF-отчёт подготовлен.');
    } catch (error) {
      if (revision === state.revision && !exportController.signal.aborted) {
        $('pdf-error').textContent = error instanceof Error ? error.message : 'Не удалось сформировать PDF. Попробуйте ещё раз.';
        $('pdf-error').hidden = false;
      }
    } finally {
      if (revision === state.revision) {
        state.exporting = false;
        state.exportController = null;
        buttons.forEach(button => { button.disabled = false; button.querySelector('span').textContent = 'Скачать PDF'; });
      }
    }
  }
  $('dropzone').addEventListener('click', () => $('video-input').click());
  $('video-input').addEventListener('change', event => selectFile(event.target.files[0]));
  $('remove-file').addEventListener('click', () => { resetSelection(); $('dropzone').focus(); });
  $('load-demo').addEventListener('click', selectDemo);
  $('demo-scenario').addEventListener('change', () => { if (state.source === 'demo') selectDemo(); });
  $('analyze-button').addEventListener('click', analyze);
  $('cancel-button').addEventListener('click', () => { invalidate(); state.phase = state.source ? 'ready' : 'idle'; renderUpload(); $('analyze-button').focus(); toast('Проверка отменена.'); });
  $('view-violations').addEventListener('click', () => { if (hasReport()) location.hash = 'report'; });
  $('download-pdf').addEventListener('click', downloadPDF);
  $('download-pdf-bottom').addEventListener('click', downloadPDF);
  $('violation-list').addEventListener('click', event => {
    const button = event.target.closest('[data-frame]');
    if (!button || !hasReport()) return;
    const index = Number(button.dataset.event), frameIndex = Number(button.dataset.frame);
    const violation = state.result.report.violations[index], frame = violation?.frames[frameIndex];
    if (!frame) return;
    $('frame-title').textContent = `Нарушение ${index + 1} · ${timecode(violation.timecode_sec)}`;
    $('frame-image').src = frame.src; $('frame-image').alt = frame.caption;
    $('frame-caption').textContent = frame.caption;
    $('frame-dialog').showModal();
  });
  $('close-frame').addEventListener('click', () => $('frame-dialog').close());
  $('frame-dialog').addEventListener('click', event => { if (event.target !== $('frame-dialog')) return; const rect = $('frame-dialog').getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('frame-dialog').close(); });
  let dragDepth = 0;
  $('dropzone').addEventListener('dragenter', event => { event.preventDefault(); if (state.phase !== 'processing') { dragDepth++; $('dropzone').classList.add('drag-over'); } });
  $('dropzone').addEventListener('dragover', event => event.preventDefault());
  $('dropzone').addEventListener('dragleave', event => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('dropzone').classList.remove('drag-over'); });
  $('dropzone').addEventListener('drop', event => { event.preventDefault(); dragDepth = 0; $('dropzone').classList.remove('drag-over'); if (state.phase !== 'processing') { const files = event.dataTransfer.files; selectFile(files[0]); if (files.length > 1) toast('Выбран первый файл. Загружайте видео по одному.'); } });
  document.addEventListener('dragover', event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });
  document.addEventListener('drop', event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });
  document.querySelector('.skip-link').addEventListener('click', event => { event.preventDefault(); $('main').focus(); });
  window.addEventListener('hashchange', () => route());
  $('demo-tools').hidden = !demoMode;
  if (demoMode) { $('mode-badge').classList.add('demo'); $('mode-badge').innerHTML = '<span></span>Демонстрация'; }
  if (client.available) $('privacy-note').textContent = 'Видео будет передано на сервер для анализа после нажатия «Проверить видео».';
  renderIcons(); renderUpload(); route(false);
})();
