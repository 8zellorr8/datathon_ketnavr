/* Local PDF export. Requires vendor/pdf-lib.min.js and vendor/fontkit.umd.min.js. */
(function (root) {
  'use strict';
  const script = typeof document !== 'undefined' ? document.currentScript : null;
  const baseURL = script && script.src ? new URL('.', script.src).href : (root.location ? root.location.href : 'http://localhost/');
  const text = value => String(value == null ? '' : value).replace(/\r\n?/g, '\n').replace(/[\u2010-\u2015]/g, '-').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '').replace(/\t/g, '    ');
  const time = value => {
    const ticks = Math.round(Math.max(0, Number(value) || 0) * 10);
    const hours = Math.floor(ticks / 36000), minutes = Math.floor(ticks / 600) % 60, seconds = Math.floor(ticks / 10) % 60;
    return `${hours ? String(hours).padStart(2, '0') + ':' : ''}${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${ticks % 10}`;
  };
  function friendly(message, cause) {
    const error = new Error(message);
    if (cause) error.cause = cause;
    return error;
  }
  async function bytes(url, label) {
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = controller ? setTimeout(() => controller.abort(), 15000) : null;
    try {
      const response = await fetch(url, { signal: controller ? controller.signal : undefined, credentials: 'same-origin' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return { bytes: new Uint8Array(await response.arrayBuffer()), type: response.headers.get('content-type') || '' };
    } catch (error) {
      throw friendly(`Не удалось загрузить ${label} для PDF. Проверьте доступность файлов сайта и повторите скачивание.`, error);
    } finally { if (timer) clearTimeout(timer); }
  }
  function validSource(src) {
    if (typeof src !== 'string' || !src) throw friendly('Для одного из кадров не указан адрес изображения.');
    if (/^data:image\/(jpeg|jpg|png|webp);base64,/i.test(src)) return src;
    const url = new URL(src, baseURL);
    const base = new URL(baseURL);
    if (url.origin !== base.origin || !['http:', 'https:', 'file:', 'blob:'].includes(url.protocol)) {
      throw friendly('Кадры для PDF должны быть локальными изображениями сайта или JPEG/PNG/WebP в base64.');
    }
    return url.href;
  }
  async function toJPEG(buffer, mime) {
    const blob = new Blob([buffer], { type: mime || 'image/png' });
    const objectURL = URL.createObjectURL(blob);
    try {
      const image = await new Promise((resolve, reject) => {
        const value = new Image();
        const timer = setTimeout(() => reject(new Error('Image timeout')), 15000);
        value.onload = () => { clearTimeout(timer); resolve(value); };
        value.onerror = () => { clearTimeout(timer); reject(new Error('Invalid image')); };
        value.src = objectURL;
      });
      const scale = Math.min(1, 1600 / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Canvas unavailable');
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.9);
    } finally { URL.revokeObjectURL(objectURL); }
  }
  async function build(report) {
    if (!report || !Array.isArray(report.violations) || !report.violations.length) {
      throw friendly('В этом отчёте нет событий для экспорта.');
    }
    if (!root.PDFLib || !root.fontkit) throw friendly('Модуль PDF ещё не загрузился. Обновите страницу и повторите скачивание.');
    const { PDFDocument, rgb, PDFName, PDFString } = root.PDFLib;
    const doc = await PDFDocument.create();
    doc.registerFontkit(root.fontkit);
    const fontFiles = await Promise.all([
      bytes(new URL('assets/fonts/NotoSans-Regular.ttf', baseURL).href, 'шрифт'),
      bytes(new URL('assets/fonts/NotoSans-Bold.ttf', baseURL).href, 'шрифт заголовков')
    ]);
    const regular = await doc.embedFont(fontFiles[0].bytes, { subset: true });
    const bold = await doc.embedFont(fontFiles[1].bytes, { subset: true });
    const colors = { ink: rgb(.12, .16, .18), muted: rgb(.39, .43, .46), accent: rgb(.62, .09, .27), line: rgb(.86, .88, .89), pale: rgb(.97, .96, .97) };
    const width = 595.28, height = 841.89, margin = 44, inner = width - margin * 2, bottom = 58;
    let page, y, continuation = '';
    const imageCache = new Map();
    doc.setTitle(report.isDemo ? 'ForteLens - демонстрационный отчёт' : 'ForteLens - отчёт о проверке');
    doc.setAuthor('ForteLens');
    doc.setCreator('ForteLens browser PDF export');
    doc.setLanguage('ru-RU');
    function addPage() {
      page = doc.addPage([width, height]);
      page.drawText('ForteLens', { x: margin, y: height - 37, size: 14, font: bold, color: colors.accent });
      const tag = report.isDemo ? 'ДЕМОНСТРАЦИОННЫЙ ОТЧЁТ' : 'ОТЧЁТ О ПРОВЕРКЕ';
      page.drawText(tag, { x: width - margin - regular.widthOfTextAtSize(tag, 8), y: height - 35, size: 8, font: regular, color: colors.muted });
      page.drawLine({ start: { x: margin, y: height - 49 }, end: { x: width - margin, y: height - 49 }, thickness: 0.6, color: colors.line });
      y = height - 76;
      if (continuation) { page.drawText(continuation, { x: margin, y, size: 9, font: regular, color: colors.muted }); y -= 23; }
    }
    function ensure(space) { if (y - space < bottom) addPage(); }
    function wrap(value, font, size, available = inner) {
      const lines = [];
      for (const paragraph of text(value).split('\n')) {
        if (!paragraph.trim()) { lines.push(''); continue; }
        let line = '';
        for (const word of paragraph.trim().split(/\s+/)) {
          const candidate = line ? `${line} ${word}` : word;
          if (font.widthOfTextAtSize(candidate, size) <= available) { line = candidate; continue; }
          if (line) { lines.push(line); line = ''; }
          if (font.widthOfTextAtSize(word, size) <= available) { line = word; continue; }
          for (const character of Array.from(word)) {
            if (line && font.widthOfTextAtSize(line + character, size) > available) { lines.push(line); line = ''; }
            line += character;
          }
        }
        if (line) lines.push(line);
      }
      return lines;
    }
    function paragraph(value, options = {}) {
      const font = options.bold ? bold : regular;
      const size = options.size || 10.5;
      const leading = options.leading || size * 1.5;
      for (const line of wrap(value, font, size, options.width || inner)) {
        ensure(leading);
        if (line) {
          page.drawText(line, { x: margin, y, size, font, color: options.color || colors.ink });
          if (options.url) {
            const annotation = doc.context.obj({ Type: PDFName.of('Annot'), Subtype: PDFName.of('Link'), Rect: [margin, y - 3, margin + font.widthOfTextAtSize(line, size), y + size], Border: [0, 0, 0], A: { Type: PDFName.of('Action'), S: PDFName.of('URI'), URI: PDFString.of(options.url) } });
            page.node.addAnnot(doc.context.register(annotation));
          }
        }
        y -= leading;
      }
      y -= options.after == null ? 9 : options.after;
    }
    function heading(value) { ensure(55); paragraph(value, { bold: true, size: 10, color: colors.accent, after: 6 }); }
    async function image(frame, eventIndex, frameIndex) {
      const source = validSource(frame.src);
      let embedded = imageCache.get(source);
      if (!embedded) {
        const payload = await bytes(source, `кадр ${frameIndex + 1} события ${eventIndex + 1}`);
        try {
          if (typeof document !== 'undefined' && typeof Image !== 'undefined') {
            let mime = payload.type;
            if (/\.svg(?:[?#]|$)/i.test(source)) mime = 'image/svg+xml';
            embedded = await doc.embedJpg(await toJPEG(payload.bytes, mime));
          } else if (payload.bytes[0] === 0xff && payload.bytes[1] === 0xd8) {
            embedded = await doc.embedJpg(payload.bytes);
          } else if (payload.bytes[0] === 0x89 && payload.bytes[1] === 0x50) {
            embedded = await doc.embedPng(payload.bytes);
          } else throw new Error('SVG conversion requires a browser');
          imageCache.set(source, embedded);
        } catch (error) { throw friendly(`Не удалось включить кадр ${frameIndex + 1} события ${eventIndex + 1} в PDF. Проверьте изображение и повторите скачивание.`, error); }
      }
      const scale = Math.min(inner / embedded.width, 225 / embedded.height);
      const w = embedded.width * scale, h = embedded.height * scale;
      ensure(h + 39);
      page.drawRectangle({ x: margin, y: y - h, width: inner, height: h, color: colors.pale });
      page.drawImage(embedded, { x: margin + (inner - w) / 2, y: y - h, width: w, height: h });
      y -= h + 15;
      paragraph(frame.caption || `Кадр ${frameIndex + 1}`, { size: 8.5, color: colors.muted, after: 10 });
    }
    addPage();
    paragraph(report.isDemo ? 'Пример отчёта о проверке' : 'Отчёт о проверке видео', { bold: true, size: 23, leading: 30, after: 8 });
    paragraph(`Файл: ${report.fileName || 'Название не указано'}`, { size: 9.5, after: 3 });
    const date = new Date(report.createdAt || Date.now());
    paragraph(`Дата: ${Number.isNaN(date.getTime()) ? text(report.createdAt) : date.toLocaleString('ru-RU')}  |  Событий: ${report.violations.length}`, { size: 9.5, color: colors.muted, after: 12 });
    if (report.isDemo) paragraph('ДЕМОНСТРАЦИЯ. События и изображения условные. Этот отчёт не является результатом ИИ-анализа загруженного видео и не подтверждает нарушение.', { size: 9, color: colors.accent, after: 18 });
    else paragraph('Отчёт содержит признаки возможных нарушений. Кадры и применимость нормативного основания требуют проверки человеком.', { size: 9, color: colors.muted, after: 18 });
    for (let index = 0; index < report.violations.length; index++) {
      const item = report.violations[index];
      if (!item || !Array.isArray(item.frames) || !item.frames.length) throw friendly(`Для события ${index + 1} отсутствуют кадры. Добавьте их перед экспортом.`);
      continuation = '';
      ensure(330);
      paragraph(`Событие ${index + 1}. ${item.title || 'Требует проверки'}`, { bold: true, size: 15, leading: 21, after: 9 });
      continuation = `Событие ${index + 1} - продолжение`;
      for (let frameIndex = 0; frameIndex < item.frames.length; frameIndex++) await image(item.frames[frameIndex], index, frameIndex);
      heading('Время события');
      paragraph(time(item.timecode_sec), { bold: true, size: 13, after: 12 });
      heading('Описание наблюдения');
      paragraph(item.description || 'Описание не указано.');
      heading('Нормативное основание');
      const legal = item.legal || {};
      paragraph(legal.act || 'Нормативный акт не указан.', { bold: true, after: 5 });
      const clause = text(legal.paragraph || '');
      paragraph(/^(пункт|п\.)/i.test(clause) ? clause : `Пункт: ${clause || 'не указан'}`, { after: 6 });
      if (legal.context) paragraph(legal.context, { size: 10 });
      if (legal.url) {
        let url;
        try { url = new URL(legal.url); } catch (error) { throw friendly(`В событии ${index + 1} указана некорректная ссылка на нормативный акт.`, error); }
        if (!['http:', 'https:'].includes(url.protocol)) throw friendly(`В событии ${index + 1} ссылка на нормативный акт должна начинаться с http или https.`);
        heading('Источник');
        paragraph(url.href, { size: 8.5, color: colors.accent, url: url.href });
      } else paragraph('Ссылка на источник не указана.', { size: 9, color: colors.muted });
      y -= 10;
    }
    const pages = doc.getPages();
    pages.forEach((current, index) => {
      current.drawLine({ start: { x: margin, y: 42 }, end: { x: width - margin, y: 42 }, thickness: 0.5, color: colors.line });
      current.drawText(report.isDemo ? 'ForteLens | Демо - не доказательство' : 'ForteLens | Для проверки человеком', { x: margin, y: 27, size: 8, font: regular, color: colors.muted });
      const number = `${index + 1} / ${pages.length}`;
      current.drawText(number, { x: width - margin - regular.widthOfTextAtSize(number, 8), y: 27, size: 8, font: regular, color: colors.muted });
    });
    return new Blob([await doc.save()], { type: 'application/pdf' });
  }
  async function download(report, { signal } = {}) {
    const checkAborted = () => {
      if (!signal?.aborted) return;
      const error = typeof DOMException !== 'undefined' ? new DOMException('Экспорт отменён.', 'AbortError') : Object.assign(new Error('Экспорт отменён.'), { name: 'AbortError' });
      throw error;
    };
    try {
      checkAborted();
      const blob = await build(report);
      // The user may select another video while fonts/images are loading.
      // Never trigger a download for an invalidated report.
      checkAborted();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      const name = text(report.fileName || 'video').replace(/\.[^.]+$/, '').replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').slice(0, 100);
      anchor.href = url;
      anchor.download = `ForteLens_${report.isDemo ? 'DEMO_' : ''}${name}.pdf`;
      anchor.hidden = true;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      return blob;
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      if (/Не удалось|не указан|отсутствуют|нет событий|Модуль PDF|Кадры для PDF|ссылка на нормативный акт|некорректная ссылка/.test(error.message || '')) throw error;
      throw friendly('Не удалось создать PDF. Повторите скачивание; если ошибка сохранится, проверьте кадры и данные отчёта.', error);
    }
  }
  root.ForteLensPDF = Object.freeze({ build, download });
})(typeof window !== 'undefined' ? window : globalThis);
