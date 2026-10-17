/* Replace this adapter when the backend is ready. No requests are made now. */
window.ForteLensAnalysis = Object.freeze({
  available: false,
  async analyze(file, { signal, onStatus } = {}) {
    if (signal?.aborted) throw new DOMException('Отменено', 'AbortError');
    throw new Error('Сервис анализа ещё не подключён.');
  }
});
