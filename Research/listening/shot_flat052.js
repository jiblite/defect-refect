// Screenshots of the listening page's newest section (its head and its last five panels), wide and on a phone, for a look.
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  for (const [name, w, scheme] of [['wide', 900, 'light'], ['phone', 390, 'dark']]) {
    const p = await b.newPage({ viewport: { width: w, height: 900 }, colorScheme: scheme, deviceScaleFactor: 1 });
    await p.goto('file:///home/claude/page_v2/index.html'); await p.waitForTimeout(800);
    const sec = await p.$('#flat052'); await sec.screenshot({ path: name + '_head.png' });
    const panels = await p.$$('section.panel'); const n = panels.length;
    for (let i = n - 5; i < n; i++) await panels[i].screenshot({ path: name + '_p' + (i - n + 6) + '.png' });
    const over = await p.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    console.log(name, 'panels', n, 'sideways scroll:', over);
    await p.close();
  }
  await b.close();
})();
