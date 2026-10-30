const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8766;
const server = spawn('python', ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', 'dist'], {stdio: 'ignore'});

(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
    const page = await browser.newPage();
    const messages = [];
    page.on('console', x => { if (x.type() === 'error') messages.push(x.text()); });
    page.on('pageerror', error => messages.push(String(error)));
    // The notebook code remains conventional Python. The kernel handles
    // tensorflow imports and the asynchronous TF.js implementation.
    const python = [
      'import tensorflow as tf',
      'from tensorflow.keras.models import Sequential',
      'from tensorflow.keras.layers import Dense',
      'model = Sequential([Dense(2, activation="sigmoid", input_shape=(2,)), Dense(1, activation="sigmoid")])',
      'model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))',
      'model.fit([[0,0],[1,0],[0,1],[1,1]], [[0],[1],[1],[1]], epochs=2)',
      'print("ML_KERNEL_ZERO_EDIT_OK")'
    ].join('\n');
    const url = `http://127.0.0.1:${port}/repl/index.html?kernel=python&code=${encodeURIComponent(python)}`;
    const response = await page.goto(url, {waitUntil:'domcontentloaded', timeout:60000});
    assert.equal(response.status(), 200, 'JupyterLite REPL must be available');
    await page.waitForFunction(() => document.body.innerText.includes('ML_KERNEL_ZERO_EDIT_OK'), null, {timeout:120000});
    console.log('Original Python TensorFlow syntax executed successfully in custom JupyterLite kernel');
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
