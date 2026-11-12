import { chromium } from 'playwright';

const browser = await chromium.launch({ channel: 'chrome', args: ['--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
page.on('console', (m) => console.log('[console]', m.type(), m.text().slice(0, 200)));
await page.goto('http://127.0.0.1:8210');
let info;
for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(2000);
    info = await page.evaluate(() => {
        const v = document.getElementById('remote-video');
        return v ? { w: v.videoWidth, h: v.videoHeight, t: v.currentTime, ready: v.readyState } : null;
    });
    if (info && info.w > 0 && info.t > 1) break;
}
console.log('VIDEO', JSON.stringify(info));
await page.screenshot({ path: 'stream.png' });
await browser.close();
