const { chromium } = require('playwright'); const fs = require('fs');
(async () => {
  const b = await chromium.launch(); const scheme = process.argv[2] || 'light'; const w = parseInt(process.argv[3] || '900');
  const p = await b.newPage({ viewport: { width: w, height: 1300 }, deviceScaleFactor: 1, colorScheme: scheme });
  await p.goto('file:///home/claude/page_v2/index.html'); await p.waitForTimeout(1200);
  const y = await p.evaluate(() => document.getElementById('cut045').getBoundingClientRect().top + window.scrollY);
  for (let i = 0; i < 4; i++) { await p.evaluate((v) => window.scrollTo(0, v), y - 20 + i * 1250); await p.waitForTimeout(300); await p.screenshot({ path: '/home/claude/page_v2/shot_cut045_' + scheme + '_' + w + '_' + (i + 1) + '.png' }); }
  const m = await p.evaluate(() => ({ h: document.documentElement.scrollHeight, audios: document.querySelectorAll('audio').length, over: document.documentElement.scrollWidth > window.innerWidth,
    srcs: Array.from(document.querySelectorAll('audio')).map(a => a.getAttribute('src')), h3: Array.from(document.querySelectorAll('h3')).map(a => a.textContent).slice(-6),
    tw: Array.from(document.querySelectorAll('#cut045 .tablewrap')).map(t => [t.scrollWidth, t.clientWidth]) }));
  const missing = m.srcs.filter(s => !fs.existsSync('/home/claude/page_v2/' + s));
  console.log(JSON.stringify({ h: m.h, audios: m.audios, over: m.over, missing, h3: m.h3, tablewrap: m.tw })); await b.close();
})();
