/* Local service adapter. The API key lives only in this page's memory. */
(() => {
  'use strict';
  const MAX_FILE_BYTES = 64 * 1024 * 1024;
  const DEADLINE_MS = 5 * 60 * 1000;
  const POLL_MS = 1000;
  const JOB_ID = /^[0-9a-f]{32}$/;
  const KNOWN_SOURCES = new Set(['https://adilet.zan.kz/rus/docs/V1900019680']);
  let apiKey = '';

  const messages = Object.freeze({
    unauthorized: 'API-ключ не принят. Проверь ключ локального сервиса и повтори запрос.',
    queue_full: 'Сервис уже обрабатывает видео, очередь заполнена. Дождись завершения и попробуй снова.',
    upload_too_large: 'Размер запроса превышает 64 МиБ. Выбери MOV немного меньше 64 МиБ: служебные поля загрузки тоже занимают место.',
    invalid_extension: 'Для анализа нужен файл с расширением .mov.',
    empty_file: 'Видео пустое. Выбери другой файл.',
    empty_upload: 'Не удалось передать видео. Выбери файл и повтори загрузку.',
    incomplete_upload: 'Видео передалось не полностью. Проверь соединение с локальным сервисом и повтори загрузку.',
    upload_timeout: 'Загрузка заняла слишком много времени. Результат анализа не получен.',
    model_out_of_memory: 'Модели не хватило оперативной памяти или памяти видеокарты. Закрой лишние приложения и повтори анализ. Результат не получен.',
    model_timeout: 'Локальная модель не успела обработать запрос. Результат не получен; попробуй снова после завершения текущей обработки.',
    model_invalid_output: 'Модель вернула некорректный ответ даже после повторной проверки. Результат не получен. Можно повторить анализ.',
    analysis_timeout: 'Истекли 5 минут обработки. Результат не получен.',
    time_budget_exceeded: 'Не хватило времени на полную проверку видео. Результат не получен.',
    candidate_budget_exceeded: 'Сервис не успел проверить все найденные события. Результат не получен.',
    evidence_budget_exceeded: 'Для проверки события требуется больше кадров, чем поддерживает текущая настройка. Результат не получен.',
    video_duration_exceeded: 'Видео длиннее поддерживаемых 180 секунд. Выбери более короткий ролик.',
    invalid_analysis_configuration: 'Настройки локального анализа некорректны. Проверь конфигурацию сервиса.',
    invalid_rule_catalogue: 'Каталог правил локального сервиса некорректен. Проверь его конфигурацию.',
    no_verified_rules: 'В сервисе не настроены проверенные нормативные правила.',
    analysis_failed: 'Локальный анализ завершился ошибкой. Результат не получен. Подробности доступны в журнале сервиса.',
    job_not_found: 'Задача не найдена или уже удалена после перезапуска сервиса. Загрузи видео заново.',
    server_stopped: 'Сервис остановился во время обработки. Запусти его и повтори загрузку.',
    server_unavailable: 'Локальный сервис временно недоступен. Проверь, что backend запущен.',
    bad_response: 'Сервис вернул некорректный или неполный результат. Вердикт не показан. Проверь журнал сервиса.'
  });

  function failure(code = 'bad_response') {
    const error = new Error(messages[code] || messages.bad_response);
    error.code = code;
    return error;
  }
  function isReady() { return apiKey.length > 0; }
  function setApiKey(value) {
    const next = typeof value === 'string' ? value.trim() : '';
    // Header values must be printable ASCII, without control characters.
    apiKey = next.length <= 512 && /^[\x21-\x7e]*$/.test(next) ? next : '';
    window.dispatchEvent(new CustomEvent('fortelens:credentials-changed'));
  }
  function text(value) {
    if (typeof value !== 'string' || !value.trim()) throw failure();
    return value.trim();
  }
  function validTime(value) { return Number.isFinite(value) && value >= 0; }
  function formatTime(value) {
    const ms = Math.round(value * 1000);
    const minutes = Math.floor(ms / 60000);
    return `${String(minutes).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0')}`;
  }
  function jpegSource(value) {
    if (typeof value !== 'string' || !value || value.length > 12 * 1024 * 1024 || value.length % 4 !== 0 ||
        !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) throw failure();
    let bytes;
    try { bytes = atob(value); } catch { throw failure(); }
    if (bytes.length < 4 || bytes.charCodeAt(0) !== 255 || bytes.charCodeAt(1) !== 216 ||
        bytes.charCodeAt(bytes.length - 2) !== 255 || bytes.charCodeAt(bytes.length - 1) !== 217) throw failure();
    return `data:image/jpeg;base64,${value}`;
  }
  function legalSource(source) {
    if (!source || source.verified !== true) return null;
    if (![source.title, source.section, source.url].every(value => typeof value === 'string' && value.trim())) return null;
    let url;
    try { url = new URL(source.url); } catch { return null; }
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password ||
        !KNOWN_SOURCES.has(url.origin + url.pathname) || url.search) return null;
    return { act: source.title.trim(), paragraph: source.section.trim(), url: url.href,
      context: 'Проверен нормативный источник. Применимость пункта к событию и итоговый вывод должен проверить специалист.' };
  }
  function normalizeResult(raw, jobId) {
    if (!raw || raw.status !== 'done' || !['violation', 'no_violation'].includes(raw.verdict) ||
        !Array.isArray(raw.violations) || (raw.verdict === 'violation') !== (raw.violations.length > 0)) throw failure();
    const duration = raw.analysis?.duration_sec;
    if (duration !== undefined && !validTime(duration)) throw failure();
    const violations = raw.violations.map((event, index) => {
      if (!event || !validTime(event.timecode_sec) || (duration !== undefined && event.timecode_sec > duration + 0.05) ||
          !Array.isArray(event.evidence_frames) || event.evidence_frames.length < 1 || event.evidence_frames.length > 3) throw failure();
      if (event.verification_status !== undefined && !['normative_source_verified', 'needs_regulation_review'].includes(event.verification_status)) throw failure();
      const legal = event.verification_status === 'needs_regulation_review' ? null : legalSource(event.rule_source);
      const regulatoryNote = typeof event.regulatory_note === 'string' && event.regulatory_note.trim() ? event.regulatory_note.trim() :
        legal ? '' : 'Визуальное наблюдение требует проверки специалистом. Применимый нормативный пункт не подтверждён.';
      return {
        id: `${jobId}:${index + 1}`, title: text(event.type), timecode_sec: event.timecode_sec,
        description: text(event.description),
        frames: event.evidence_frames.map((frame, frameIndex) => {
          if (!frame || !validTime(frame.timestamp_sec) || (duration !== undefined && frame.timestamp_sec > duration + 0.05)) throw failure();
          return { src: jpegSource(frame.image_base64), timestamp_sec: frame.timestamp_sec,
            caption: `Кадр ${frameIndex + 1} · ${formatTime(frame.timestamp_sec)}` };
        }),
        legal, verification_status: legal ? 'normative_source_verified' : 'needs_regulation_review',
        regulatory_note: regulatoryNote, requires_human_review: true
      };
    });
    return { verdict: raw.verdict, violations, job_id: jobId,
      analysis: { coverage: 'configured_checks_only',
        ...(validTime(duration) ? { duration_sec: duration } : {}),
        ...(validTime(raw.analysis?.elapsed_sec) ? { elapsed_sec: raw.analysis.elapsed_sec } : {}) } };
  }
  async function readResponse(response, expectedStatus, signal) {
    let payload;
    try { payload = await response.json(); } catch (error) {
      if (signal.aborted) throw error;
      throw failure();
    }
    if (!response.ok) {
      const code = payload?.error?.code;
      if (Object.hasOwn(messages, code)) throw failure(code);
      if (response.status === 401 || response.status === 403) throw failure('unauthorized');
      if (response.status === 413) throw failure('upload_too_large');
      if (response.status === 429) throw failure('queue_full');
      throw failure('server_unavailable');
    }
    if (response.status !== expectedStatus || !payload || typeof payload !== 'object') throw failure();
    return payload;
  }
  function pause(signal) {
    return new Promise((resolve, reject) => {
      if (signal.aborted) { reject(new DOMException('Ожидание прекращено.', 'AbortError')); return; }
      const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(new DOMException('Ожидание прекращено.', 'AbortError')); };
      const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, POLL_MS);
      signal.addEventListener('abort', abort, { once: true });
    });
  }
  async function analyze(file, { signal, onStatus } = {}) {
    if (signal?.aborted) throw new DOMException('Ожидание прекращено.', 'AbortError');
    if (!file || typeof file.name !== 'string' || !/\.mov$/i.test(file.name)) throw failure('invalid_extension');
    if (!Number.isFinite(file.size) || file.size <= 0) throw failure('empty_file');
    if (file.size >= MAX_FILE_BYTES) throw failure('upload_too_large');
    if (!isReady()) throw failure('unauthorized');
    const requestKey = apiKey;
    const controller = new AbortController();
    let deadlineExpired = false;
    const started = performance.now();
    const timer = setTimeout(() => { deadlineExpired = true; controller.abort(); }, DEADLINE_MS);
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    const reportStatus = message => { if (typeof onStatus === 'function') onStatus(message); };
    const request = (url, options = {}) => fetch(url, { ...options, signal: controller.signal,
      headers: { 'X-API-Key': requestKey }, credentials: 'omit', cache: 'no-store', mode: 'same-origin', redirect: 'error' });
    try {
      const data = new FormData();
      data.append('file', file, file.name);
      reportStatus('Передаём видео в локальный сервис. Загрузка и анализ могут занять до 5 минут.');
      const submitted = await readResponse(await request('/api/v1/analyze', { method: 'POST', body: data }), 202, controller.signal);
      if (!JOB_ID.test(submitted.job_id || '') || submitted.status !== 'processing') throw failure();
      const jobId = submitted.job_id;
      while (true) {
        if (performance.now() - started >= DEADLINE_MS) { deadlineExpired = true; controller.abort(); throw failure('analysis_timeout'); }
        reportStatus('Локальная модель проверяет видео. Ожидание включает очередь; результат появится после полной обработки.');
        const job = await readResponse(await request(`/api/v1/jobs/${jobId}`), 200, controller.signal);
        if (job.job_id !== jobId || !['processing', 'done', 'failed'].includes(job.status)) throw failure();
        if (job.status === 'done') return normalizeResult(job, jobId);
        if (job.status === 'failed') {
          if (job.verdict !== undefined || job.violations !== undefined) throw failure();
          throw failure(Object.hasOwn(messages, job.error?.code) ? job.error.code : 'analysis_failed');
        }
        if (job.verdict !== undefined || job.violations !== undefined) throw failure();
        await pause(controller.signal);
      }
    } catch (error) {
      if (signal?.aborted) throw new DOMException('Ожидание прекращено. Задача на сервере может продолжаться.', 'AbortError');
      if (deadlineExpired) throw failure('analysis_timeout');
      if (error instanceof Error && Object.hasOwn(messages, error.code)) throw error;
      throw failure('server_unavailable');
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
    }
  }
  function bindCredentials() {
    const input = document.getElementById('api-key');
    const toggle = document.getElementById('toggle-api-key');
    if (!input || !toggle) return;
    input.addEventListener('input', () => setApiKey(input.value));
    toggle.addEventListener('click', () => {
      const visible = input.type === 'password';
      input.type = visible ? 'text' : 'password';
      toggle.textContent = visible ? 'Скрыть' : 'Показать';
      toggle.setAttribute('aria-pressed', String(visible));
      toggle.setAttribute('aria-label', visible ? 'Скрыть API-ключ' : 'Показать API-ключ');
    });
    setApiKey(input.value);
  }
  window.ForteLensAnalysis = Object.freeze({ available: true, isReady, setApiKey, analyze });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bindCredentials, { once: true });
  else bindCredentials();
})();
