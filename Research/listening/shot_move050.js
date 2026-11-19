// Screenshots of the page's 0.5.0 section (light and dark, wide and phone), with a pointer over the first curve.
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const browser = await chromium.launch(); const url = 'file://' + path.resolve(__dirname, 'index.html'); const out = [];
  for (const [scheme, width] of [['light', 900], ['dark', 900], ['dark', 390]]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme, deviceScaleFactor: 1 }); const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    await page.goto(url); await page.waitForTimeout(600);
    const sec = page.locator('#move050'); await sec.scrollIntoViewIfNeeded(); await sec.screenshot({ path: `shot_m5_${scheme}_${width}_head.png` });
    const panels = page.locator('section.panel'); const n = await panels.count();
    for (const k of [40, 42, 44, 46]) { const p = page.locator('section.panel', { has: page.locator('h3', { hasText: new RegExp('^' + k + '\\.') }) }); await p.scrollIntoViewIfNeeded(); if (k === 40) { const svg = p.locator('.plot svg'); const b = await svg.boundingBox(); await page.mouse.move(b.x + b.width * 0.62, b.y + b.height * 0.5); await page.waitForTimeout(150); } await p.screenshot({ path: `shot_m5_${scheme}_${width}_${k}.png` }); }
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    out.push(`${scheme} ${width}: panels ${n}, sideways overflow ${over}px, errors ${errors.length ? errors.join(' | ').slice(0, 300) : 'none'}`); await page.close();
  }
  console.log(out.join('\n')); await browser.close();
})();
