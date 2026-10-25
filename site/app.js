(() => {
  const base = document.documentElement.dataset.base;
  const dialog = document.getElementById('search-dialog');
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');
  const status = document.getElementById('search-status');
  let index;
  document.querySelectorAll('[data-open-search]').forEach(button => button.addEventListener('click', () => {
    dialog.showModal(); input.focus();
  }));
  document.getElementById('search-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  input.addEventListener('input', async () => {
    const query = input.value.trim().toLocaleLowerCase();
    results.replaceChildren();
    if (query.length < 2) { status.textContent = '输入至少两个字或字母，搜索全书正文。'; return; }
    try {
      if (!index) { const response = await fetch(base + 'search.json'); if (!response.ok) throw new Error('fetch'); index = await response.json(); }
      if (input.value.trim().toLocaleLowerCase() !== query) return;
      const matches = index.filter(item => (item.title + ' ' + item.text).toLocaleLowerCase().includes(query));
      status.textContent = `找到 ${matches.length} 个章节` + (matches.length > 25 ? '（显示前25个）' : '');
      matches.slice(0, 25).forEach(item => {
        const anchor = document.createElement('a'); anchor.className = 'search-hit'; anchor.href = base + item.url;
        const title = document.createElement('b'); title.textContent = item.title;
        const snippet = document.createElement('p'); const at = item.text.toLocaleLowerCase().indexOf(query);
        const start = Math.max(0, at - 45); snippet.textContent = (start ? '…' : '') + item.text.slice(start, start + 155) + '…';
        anchor.append(title, snippet); results.append(anchor);
      });
    } catch { status.textContent = '搜索索引暂时无法加载，请从目录阅读。'; }
  });
  document.querySelectorAll('[data-font]').forEach(button => button.addEventListener('click', () => {
    document.documentElement.style.setProperty('--reading-size', button.dataset.font + 'px');
    try { localStorage.setItem('gpu-reading-size', button.dataset.font); } catch {}
  }));
  try {
    const size = localStorage.getItem('gpu-reading-size');
    if (['17','18','20','22'].includes(size)) document.documentElement.style.setProperty('--reading-size', size + 'px');
    const article = document.querySelector('article[data-page]');
    if (article) localStorage.setItem('gpu-last-page', article.dataset.page);
    const last = localStorage.getItem('gpu-last-page');
    const resume = document.getElementById('resume-reading');
    if (resume && last && /^[a-z0-9-]+\/$/.test(last)) { resume.href = base + last; resume.hidden = false; }
  } catch {}
})();
