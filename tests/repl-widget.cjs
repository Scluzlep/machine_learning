const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8772;
const root = path.join(process.cwd(), '.site-test-root');
const mount = path.join(root, 'machine_learning');
fs.mkdirSync(root, {recursive: true});
if (!fs.existsSync(mount)) fs.symlinkSync(path.join(process.cwd(), 'dist'), mount, 'dir');
const server = spawn('python', ['-m', 'http.server', String(port),
    '--bind', '127.0.0.1', '--directory', root], {stdio:'ignore'});

(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless: true, args:['--no-sandbox']});
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', err => errors.push(String(err)));
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    const python = [
      'import matplotlib.pyplot as plt',
      "%matplotlib widget",
      'fig, ax = plt.subplots()',
      'ax.plot([1,2,3], [1,4,9])',
      'fig.canvas.mpl_connect("button_press_event", lambda event: None)',
      'assert fig.canvas.__class__.__module__.startswith("ipympl.")',
      'print("ML_WIDGET_BACKEND_OK", plt.get_backend(), fig.canvas.__class__.__module__)',
      'plt.show()'
    ].join('\n');
    const url = `http://127.0.0.1:${port}/machine_learning/repl/index.html?kernel=python&code=${encodeURIComponent(python)}`;
    const response = await page.goto(url, {waitUntil:'domcontentloaded', timeout:60000});
    assert.equal(response.status(), 200);
    await page.waitForFunction(
      () => [...document.querySelectorAll('.jp-OutputArea-output')]
          .some(x => x.textContent.includes('ML_WIDGET_BACKEND_OK')),
      null, {timeout: 180000}
    );
    const outputs = await page.locator('.jp-OutputArea-output').allInnerTexts();
    assert.match(outputs.join('\n'), /ML_WIDGET_BACKEND_OK.*ipympl/i);
    assert.doesNotMatch(outputs.join('\n'), /not a recognised.*backend|Traceback|ModuleNotFoundError/);
    console.log('Original %matplotlib widget magic: interactive ipympl backend enabled PASS');
  } catch(error) {
    console.error(error);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => {console.error(error);process.exitCode = 1});
