/* Isolated fixture, visible only at ?demo=1. Never analyses user files. */
window.ForteLensDemo = Object.freeze({
  fileName: 'Учебный пример — кассовая зона',
  async analyze(scenario, { signal, onStatus } = {}) {
    const pause = (ms) => new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(new DOMException('Отменено', 'AbortError'));
      const onAbort = () => { clearTimeout(timer); reject(new DOMException('Отменено', 'AbortError')); };
      const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve(); }, ms);
      signal?.addEventListener('abort', onAbort, { once: true });
    });
    onStatus?.('Показываем этап проверки учебного примера…');
    await pause(700);
    onStatus?.('Подготавливаем демонстрационный результат…');
    await pause(700);
    if (scenario === 'error') throw new Error('Демонстрационная ошибка обработки. Попробуйте запустить проверку ещё раз.');
    if (scenario === 'no_violation') return { verdict: 'no_violation', violations: [] };
    return { verdict: 'violation', violations: [
      { id: 'demo-bag', title: 'Проверка инкассаторской сумки', timecode_sec: 44.2,
        description: 'На учебной схеме отмечена зона передачи сумки. В рабочем отчёте здесь будет краткое описание действия, которое модель обнаружила на записи.',
        frames: [{ src: 'assets/cash-desk.svg', caption: 'Учебная схема · зона A · не кадр видеозаписи' }],
        legal: { act: 'Постановление Правления НБРК от 29.11.2019 № 231', paragraph: 'Пункт 107', url: 'https://adilet.zan.kz/rus/docs/V1900019680', context: 'Пример связи с источником. Пункт относится к проверкам при приёме сумки инкассатором. Наличие повреждения само по себе не доказывает, что проверка не была выполнена.' } },
      { id: 'demo-room', title: 'Проверка кассовой зоны', timecode_sec: 76.8,
        description: 'На учебной схеме отмечен вход в кассовую зону. В рабочем отчёте здесь будет конкретное наблюдаемое событие и соответствующие ему кадры.',
        frames: [{ src: 'assets/cash-desk.svg', caption: 'Учебная схема · зона B · не кадр видеозаписи' }],
        legal: { act: 'Постановление Правления НБРК от 24.02.2020 № 14', paragraph: 'Пункт 8', url: 'https://adilet.zan.kz/rus/docs/V2000020075', context: 'Пример связи с источником. Пункт относится к устройству и изоляции кассового узла. Сам по себе кадр с открытой дверью не устанавливает нарушение этого пункта.' } }
    ] };
  }
});
