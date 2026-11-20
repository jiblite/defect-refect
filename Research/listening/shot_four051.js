// Screenshots of the page's 0.5.1 section (light and dark, wide and phone): its head and its four examples, the fold-outs open.
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const browser = await chromium.launch(); const url = 'file://' + path.resolve(__dirname, 'index.html'); const out = [];
  for (const [scheme, width] of [['light', 900], ['dark', 900], ['dark', 390]]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme, deviceScaleFactor: 1 }); const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    await page.goto(url); await page.waitForTimeout(600);
    const sec = page.locator('#four051'); await sec.scrollIntoViewIfNeeded(); await sec.screenshot({ path: path.resolve(__dirname, `shot_f1_${scheme}_${width}_head.png`) });
    const panels = page.locator('section.panel'); const n = await panels.count();
    for (const k of [48, 49, 50, 51]) {
      const p = page.locator('section.panel', { has: page.locator('h3', { hasText: new RegExp('^' + k + '\\.') }) }); await p.scrollIntoViewIfNeeded();
      await p.locator('details').evaluateAll(ds => ds.forEach(d => { d.open = true; })); await p.screenshot({ path: path.resolve(__dirname, `shot_f1_${scheme}_${width}_${k}.png`) });
    }
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    const table = await page.evaluate(() => { const w = document.querySelector('#four051 .tablewrap'); const t = w.querySelector('table'); return [w.clientWidth, t.scrollWidth]; });
    out.push(`${scheme} ${width}: panels ${n}, sideways overflow ${over}px, table ${table[1]}px in a frame of ${table[0]}px, errors ${errors.length ? errors.join(' | ').slice(0, 300) : 'none'}`); await page.close();
  }
  console.log(out.join('\n')); await browser.close();
})();
