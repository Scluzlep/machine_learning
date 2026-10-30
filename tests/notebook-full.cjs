const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.cwd() + '/.browser-test/node_modules/playwright');

const source = path.resolve('machine_learning');
const reportDir = path.resolve('notebook-validation');
const shard = Number(process.env.NOTEBOOK_SHARD || 0);
const shards = Number(process.env.NOTEBOOK_SHARDS || 1);
const timeout = Number(process.env.NOTEBOOK_TIMEOUT_MS || 900000);
const headed = process.env.NOTEBOOK_HEADED === '1';
assert(Number.isInteger(shard) && Number.isInteger(shards) && shards > 0 && shard >= 0 && shard < shards);

function notebooks(dir, root = dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return notebooks(full, root);
    return entry.isFile() && entry.name.endsWith('.ipynb')
      ? [path.relative(root, full).split(path.sep).join('/')]
      : [];
  });
}

const all = notebooks(source).sort();
assert(all.length >= 44, `Expected all course notebooks, found ${all.length}`);
const selected = all.filter((notebook, i) => i % shards === shard &&
  (!process.env.NOTEBOOK_FILTER || notebook === process.env.NOTEBOOK_FILTER));
assert(selected.length, 'No notebooks selected');
fs.mkdirSync(reportDir, { recursive: true });

const port = 8790;
const siteUrl = process.env.NOTEBOOK_SITE_URL || `http://127.0.0.1:${port}/machine_learning/lab/`;
let server;
if (!process.env.NOTEBOOK_SITE_URL) {
  assert(fs.existsSync(path.resolve('dist/lab/index.html')), 'Build dist/ before local validation');
  const root = path.resolve('.site-test-root');
  const mount = path.join(root, 'machine_learning');
  fs.mkdirSync(root, { recursive: true });
  if (!fs.existsSync(mount)) fs.symlinkSync(path.resolve('dist'), mount, 'dir');
  server = spawn('python', ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', root], { stdio: 'ignore' });
}

async function waitForSite() {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (server?.exitCode != null) throw new Error(`Local HTTP server exited with code ${server.exitCode}`);
    try {
      const response = await fetch(siteUrl, { signal: AbortSignal.timeout(2000) });
      if (response.ok) return;
    } catch { /* The server may still be starting. */ }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Site did not become ready: ${siteUrl}`);
}

async function runNotebook(browser, relativePath) {
  const started = Date.now();
  const result = { notebook: relativePath, status: 'failed', durationMs: 0, cells: [], error: null };
  if (relativePath.includes('切勿运行')) {
    result.status = 'excluded';
    result.error = 'Notebook filename explicitly says not to run';
    return result;
  }
  const context = await browser.newContext();
  const page = await context.newPage();
  try {
    const response = await page.goto(siteUrl, {
      waitUntil: 'domcontentloaded', timeout: 60000,
    });
    assert.equal(response.status(), 200);
    for (const component of relativePath.split('/')) {
      await page.locator('.jp-DirListing-item').filter({ hasText: component }).first().dblclick({ timeout: 60000 });
    }
    await page.locator('.jp-NotebookPanel').first().waitFor({ timeout: 60000 });
    await page.waitForFunction(() => {
      const toolbar = document.querySelector('.jp-NotebookPanel .jp-Toolbar');
      const text = toolbar?.textContent || '';
      return /Python|Pyodide/i.test(text) && !/No Kernel/i.test(text);
    }, null, { timeout: 60000 });

    let timer;
    try {
      result.cells = await Promise.race([
        page.evaluate(async () => {
        const app = window.jupyterapp;
        if (!app?.commands || !app?.shell?.currentWidget?.content?.model?.cells) {
          throw new Error('JupyterLab application API unavailable after notebook opened');
        }
        await app.commands.execute('notebook:clear-all-cell-outputs');
        await app.commands.execute('notebook:run-all-cells');
        const panel = app.shell.currentWidget;
        const cells = panel.content.model.cells;
        const summary = [];
        for (let i = 0; i < cells.length; i++) {
          const cell = cells.get(i).toJSON();
          if (cell.cell_type !== 'code' || !(Array.isArray(cell.source) ? cell.source.join('') : cell.source).trim()) continue;
          const errors = (cell.outputs || []).filter(output => output.output_type === 'error');
          summary.push({
            index: i + 1,
            executionCount: cell.execution_count ?? null,
            status: errors.length ? 'error' : cell.execution_count == null ? 'not-run' : 'passed',
            errors: errors.map(output => ({ ename: output.ename, evalue: output.evalue, traceback: output.traceback })),
          });
        }
        return summary;
        }),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Notebook timed out after ${timeout} ms`)), timeout); }),
      ]);
    } finally {
      clearTimeout(timer);
    }
    result.status = result.cells.every(cell => cell.status === 'passed') ? 'passed' : 'failed';
    if (result.status === 'failed') result.error = 'One or more code cells failed or were not executed';
  } catch (error) {
    result.error = String(error.stack || error);
  } finally {
    result.durationMs = Date.now() - started;
    await context.close();
  }
  return result;
}

(async () => {
  let browser;
  const results = [];
  try {
    await waitForSite();
    browser = await chromium.launch({ headless: !headed, args: ['--no-sandbox'] });
    for (const notebook of selected) {
      const result = await runNotebook(browser, notebook);
      results.push(result);
      console.log(`${result.status.toUpperCase()}: ${notebook} (${result.durationMs} ms)`);
      if (result.error) console.log(result.error.slice(0, 1200));
      fs.writeFileSync(path.join(reportDir, `shard-${shard}.json`), JSON.stringify({ shard, shards, discovered: all.length, results }, null, 2));
    }
  } finally {
    if (browser) await browser.close();
    if (server) server.kill();
  }
  const failed = results.filter(result => result.status === 'failed');
  console.log(JSON.stringify({ discovered: all.length, selected: selected.length, passed: results.filter(x => x.status === 'passed').length, excluded: results.filter(x => x.status === 'excluded').length, failed: failed.length }));
  if (failed.length) process.exitCode = 1;
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
  if (server) server.kill();
});

