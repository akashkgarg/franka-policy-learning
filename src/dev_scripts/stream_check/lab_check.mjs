import { chromium } from 'playwright';

// Screenshots every Franka LLM Lab view: node lab_check.mjs [url] [episode id, e.g. vision/vision_seed100000]
const url = process.argv[2] || 'http://127.0.0.1:8240';
const episode = process.argv[3];
const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
page.on('console', (m) => console.log('[console]', m.type(), m.text().slice(0, 300)));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(url);
await page.waitForSelector('#plot3d .main-svg', { timeout: 60000 });
if (episode) {
    await page.evaluate((id) => openEpisode(id), episode);
    await page.waitForFunction((id) => state.current && state.current.summary.id === id, episode, { timeout: 120000 });
    await page.waitForSelector('#plot3d .main-svg', { timeout: 60000 });
}
await page.waitForTimeout(1500);
await page.evaluate(async () => {
    const v = document.getElementById('video');
    if (v) { v.currentTime = 3.5; await new Promise((r) => setTimeout(r, 800)); }
});
await page.screenshot({ path: 'lab_episode.png', fullPage: true });
await page.click('.tab[data-view=overview]');
await page.waitForSelector('#ov-xy .main-svg', { timeout: 60000 });
await page.waitForTimeout(1000);
await page.screenshot({ path: 'lab_overview.png', fullPage: true });
await page.click('.tab[data-view=run]');
await page.waitForTimeout(2000);
await page.click('[data-watch]');
await page.waitForTimeout(2000);
await page.screenshot({ path: 'lab_run.png', fullPage: true });
await browser.close();
