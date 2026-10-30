// Execute every course notebook in a real JupyterLite Chromium session.
// Each notebook gets a clean browser context and its own JSON report/log.
// Errors are diagnostic only; never use process exit status to gate deployment.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.cwd() + '/.browser-ci/node_modules/playwright');

const suite = process.env.NOTEBOOK_SUITE || '1. Linear Regression';
const base = (process.env.SITE_URL || 'https://scluzlep.github.io/machine_learning/').replace(/\/?$/, '/');
const root = path.join(process.cwd(), 'notebook-report');
const timeoutMs = Number(process.env.NOTEBOOK_TIMEOUT_MS || 150000);
const maxCellText = 200000;
const results = [];
let browser;
fs.mkdirSync(path.join(root, 'notebooks'), {recursive: true});
fs.mkdirSync(path.join(root, 'screenshots'), {recursive: true});
const timestamp = () => new Date().toISOString();
const errorText = err => err && err.stack || String(err);
const log = (tag, value) => console.log('[' + timestamp() + '] [' + tag + '] ' + String(value));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const append = (file, object) => fs.appendFileSync(file, JSON.stringify({at:timestamp(), ...object}) + '\n');
const key = name => crypto.createHash('sha1').update(name).digest('hex').slice(0,12);
const text = value => Array.isArray(value) ? value.join('') : String(value || '');

function notebooksForSuite() {
  const rootPath = path.join(process.cwd(), 'machine_learning');
  if (suite === 'self-check') return ['浏览器环境自检.ipynb'];
  const dir = path.join(rootPath, suite);
  if (!fs.existsSync(dir)) throw Error('Notebook suite does not exist: ' + suite);
  const walk = (directory, result) => {
    for (const e of fs.readdirSync(directory,{withFileTypes:true})) {
      const abs = path.join(directory,e.name);
      if (e.isDirectory()) walk(abs,result);
      else if (e.name.endsWith('.ipynb')) result.push(path.relative(rootPath,abs).split(path.sep).join('/'));
    }
  };
  const files=[];
  walk(dir,files);
  return files.sort((a,b)=>a.localeCompare(b,'zh-CN',{numeric:true}));
}
function fromNotebook(file) {
  const data = JSON.parse(fs.readFileSync(path.join('machine_learning',file),'utf8'));
  const code = (data.cells || []).map((c,index) => ({
    cellIndex:index,
    source:text(c.source),
    type:c.cell_type
  })).filter(c=>c.type==='code');
  return {code, cellCount:(data.cells||[]).length};
}
function buildUrl(file) {
  return new URL('lab/tree/' + file.split('/').map(encodeURIComponent).join('/'),base).href;
}
async function collect(page,source,tracking) {
  const dom = await page.evaluate(() => {
    const cells=[...document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell')];
    return cells.map((cell,index)=>({
      index,
      prompt:(cell.querySelector('.jp-InputPrompt')?.textContent||'').trim(),
      busy:/\[\s*\*\s*\]/.test(cell.querySelector('.jp-InputPrompt')?.textContent||''),
      outputs:[...cell.querySelectorAll('.jp-OutputArea-output')].map(out=>({
        mime:out.getAttribute('data-mime-type'),
        text:(out.innerText||out.textContent||'').slice(0,200000),
        htmlLength:(out.innerHTML||'').length,
        hasError:!!out.querySelector('.jp-RenderedError,.jp-error,[data-mime-type="application/vnd.jupyter.error+json"]')
      }))
    }));
  }).catch(err=>({readError:errorText(err)}));
  if(!Array.isArray(dom)) return {readError:dom};
  const watched=tracking?.seen||[];
  const cells=source.code.map((cell,index)=>{
    const ui=dom[index];
    const output=ui?.outputs||[];
    const combined=output.map(x=>x.text).join('\n');
    const hasError=output.some(x=>x.hasError) ||
      /Traceback \(most recent call last\)|(?:ModuleNotFoundError|ImportError|NameError|AttributeError|TypeError|ValueError|NotImplementedError|RuntimeError|OSError|FileNotFoundError|AssertionError|SyntaxError):/i.test(combined);
    const visited=watched.includes(index);
    return {cellIndex:cell.cellIndex,codeIndex:index,source:cell.source.slice(0,100000),
      status: hasError?'FAIL':visited?'PASS':'NOT_EXECUTED',
      prompt:ui?.prompt||null,seenNewExecution:visited,
      outputs:output,hasError};
  });
  return {cells,renderedCodeCells:dom.length,sourceCodeCells:source.code.length,
    pass:cells.filter(x=>x.status==='PASS').length,
    fail:cells.filter(x=>x.status==='FAIL').length,
    unexecuted:cells.filter(x=>x.status==='NOT_EXECUTED').length};
}

async function openNotebook(page,rel) {
  const url=buildUrl(rel);
  await page.goto(url,{waitUntil:'domcontentloaded',timeout:55000});
  const panel=page.locator('.jp-NotebookPanel').first();
  try { await panel.waitFor({timeout:45000}); }
  catch (error) {
    // Fallback to the JupyterLab file browser for builds not supporting lab/tree.
    log('INFO','Direct notebook route not ready: ' + rel + ', trying file browser');
    await page.goto(new URL('lab/',base).href,{waitUntil:'domcontentloaded',timeout:45000});
    const segments=rel.split('/');
    for(const segment of segments) {
      await page.locator('.jp-DirListing-item').filter({hasText:segment}).first().dblclick({timeout:30000});
    }
    await panel.waitFor({timeout:55000});
  }
  await page.waitForFunction(()=>{
    const toolbar=document.querySelector('.jp-NotebookPanel .jp-Toolbar');
    return !!toolbar && /Python|Pyodide/i.test(toolbar.textContent||'') && !/No Kernel/i.test(toolbar.textContent||'');
  },null,{timeout:60000});
  return panel;
}
async function installTracker(page) {
  await page.evaluate(() => {
    const panel=document.querySelector('.jp-NotebookPanel');
    if(!panel) throw Error('Missing notebook panel');
    const cells=[...panel.querySelectorAll('.jp-CodeCell')];
    const state={seen:[],seenBusy:[],events:[],changed:[],initial:cells.map(c=>c.querySelector('.jp-InputPrompt')?.textContent||'')};
    const visited=new Set(),busy=new Set(),changed=new Set();
    function scan() {
      cells.forEach((cell,index)=>{
        const prompt=(cell.querySelector('.jp-InputPrompt')?.textContent||'').trim();
        if(/\[\s*\*\s*\]/.test(prompt))busy.add(index);
        if(prompt!==state.initial[index])changed.add(index);
        if(busy.has(index)||changed.has(index))visited.add(index);
      });
      state.seen=[...visited];
      state.seenBusy=[...busy];
      state.changed=[...changed];
    }
    const observer=new MutationObserver(scan);
    observer.observe(panel,{subtree:true,childList:true,characterData:true,attributes:true});
    window.__mlNotebookTrack={state,observer};
    scan();
  });
}
async function clickRunAll(page) {
  await page.locator('#jp-mainmenu-run').click({timeout:15000});
  const labels=page.locator('.lm-Menu-itemLabel');
  const count=await labels.count();
  const expected=/^(?:Run All Cells|Run All|运行所有单元格|运行全部单元格|全部运行|运行所有)$/i;
  for(let i=0;i<count;i++) {
    const label=(await labels.nth(i).innerText().catch(()=>'')).trim();
    if(expected.test(label)) {
      await labels.nth(i).click({timeout:10000});
      return label;
    }
  }
  const menu=await labels.allInnerTexts();
  throw Error('Run All Cells menu item missing; found: '+JSON.stringify(menu));
}
async function recordNotebook(rel) {
  const source=fromNotebook(rel);
  const id=key(rel);
  const filepath=path.join(root,'notebooks',id+'.json');
  const logs=path.join(root,'notebooks',id+'.jsonl');
  const record={path:rel,suite,id,started:timestamp(),sourceCodeCells:source.code.length,
    totalCells:source.cellCount,status:'FAIL',errors:[],consoleErrors:[],httpFailures:[],progress:[]};
  if(rel.includes('切勿运行')) {
    record.status='SKIP';
    record.reason='Notebook explicitly marked 切勿运行';
    fs.writeFileSync(filepath,JSON.stringify(record,null,2));
    return record;
  }
  let context,page;
  const start=Date.now();
  try {
    context=await browser.newContext({viewport:{width:1440,height:900}});
    page=await context.newPage();
    page.on('console',m=>{
      const e={kind:'console',type:m.type(),text:m.text(),location:m.location()};
      append(logs,e);
      if(m.type()==='error') record.consoleErrors.push(e);
    });
    page.on('pageerror',e=>{
      const item={kind:'pageerror',message:errorText(e)};
      append(logs,item);record.consoleErrors.push(item);
    });
    page.on('response',r=>{
      if(r.status()>=400) {
        const item={kind:'HTTP',status:r.status(),url:r.url()};
        record.httpFailures.push(item);append(logs,item);
      }
    });
    page.on('requestfailed',r=>{
      const item={kind:'requestfailed',url:r.url(),message:r.failure()?.errorText};
      record.httpFailures.push(item);append(logs,item);
    });
    page.on('worker',w=>append(logs,{kind:'worker',url:w.url()}));
    await openNotebook(page,rel);
    record.opened=timestamp();
    const actual=await page.locator('.jp-NotebookPanel .jp-CodeCell').count();
    if(actual!==source.code.length) {
      record.errors.push('Code cell count mismatch: UI='+actual+', source='+source.code.length);
    }
    if(source.code.length===0) {
      record.status='PASS';
      record.reason='Markdown-only Notebook rendered';
    } else {
      await installTracker(page);
      record.runCommand=await clickRunAll(page);
      record.runStarted=timestamp();
      const end=Date.now()+timeoutMs;
      let finished=false, previous='', stable=0;
      while(Date.now()<end) {
        await sleep(1200);
        const snapshot=await page.evaluate(()=>{
          const obj=window.__mlNotebookTrack?.state;
          const cells=[...document.querySelectorAll('.jp-NotebookPanel .jp-CodeCell')];
          const active=cells.some(c=>/\[\s*\*\s*\]/.test(c.querySelector('.jp-InputPrompt')?.textContent||''));
          const kernelText=document.querySelector('.jp-NotebookPanel .jp-Toolbar')?.textContent||'';
          return {seen:obj?.seen||[],seenBusy:obj?.seenBusy||[],changed:obj?.changed||[],
            active,kernelText,uiCodeCells:cells.length};
        });
        const compact=JSON.stringify({seen:snapshot.seen,active:snapshot.active,ui:snapshot.uiCodeCells});
        if(compact!==previous) {
          previous=compact;
          append(logs,{kind:'execution-progress',...snapshot});
          record.progress.push({at:timestamp(),...snapshot});
          log('PROGRESS',rel+' '+compact);
        }
        if(snapshot.seen.length>=source.code.length && !snapshot.active) stable++;
        else stable=0;
        if(stable>=3){finished=true;break;}
      }
      const tracking=await page.evaluate(()=>window.__mlNotebookTrack?.state||null);
      record.execution=await collect(page,source,tracking);
      record.finished=finished;
      if(!finished) record.errors.push('Timed out waiting for ALL code cells to execute ('+timeoutMs+'ms)');
      if(record.execution.fail) record.errors.push(record.execution.fail+' code cell(s) have Python error output');
      if(record.execution.unexecuted) record.errors.push(record.execution.unexecuted+' code cell(s) not observed executing');
      if(record.consoleErrors.length) record.errors.push(record.consoleErrors.length+' browser console error(s)');
      if(record.httpFailures.length) record.errors.push(record.httpFailures.length+' failed HTTP request(s)');
      record.status=record.errors.length?'FAIL':'PASS';
    }
  }catch(e) {
    record.errors.push(errorText(e));
    record.status='FAIL';
    if(page&&!page.isClosed()) {
      try {
        const tracking=await page.evaluate(()=>window.__mlNotebookTrack?.state||null);
        record.execution=await collect(page,source,tracking);
      }catch(_){}
    }
  }finally {
    record.durationSeconds=Math.round((Date.now()-start)/100)/10;
    record.finishedAt=timestamp();
    if(record.status!=='PASS' && page&&!page.isClosed()) {
      await page.screenshot({path:path.join(root,'screenshots',id+'.png'),fullPage:true,timeout:12000}).catch(()=>{});
    }
    if(context) await context.close().catch(()=>{});
    fs.writeFileSync(filepath,JSON.stringify(record,null,2));
  }
  return record;
}
function summary() {
  const counts={PASS:0,FAIL:0,SKIP:0};
  for(const r of results) counts[r.status]=(counts[r.status]||0)+1;
  const markdown=['# JupyterLite all-Notebook execution diagnostics','',
    'Suite: '+suite,'',
    '**'+results.length+' notebooks: '+counts.PASS+' PASS / '+counts.FAIL+' FAIL / '+counts.SKIP+' SKIP**',
    '',
    '| Notebook | Status | Code cells | Executed | Python errors | Seconds |',
    '| --- | --- | ---: | ---: | ---: | ---: |',
    ...results.map(r=>'| '+r.path.replace(/\|/g,'\\|')+' | '+r.status+' | '+
      r.sourceCodeCells+' | '+(r.execution?.pass||0)+' | '+(r.execution?.fail||0)+
      ' | '+(r.durationSeconds||0)+' |'),'',
    'Each notebook has its own JSON cell-output report and JSONL browser/network log. FAIL does not affect Action status.'
  ].join('\n')+'\n';
  fs.writeFileSync(path.join(root,'results.json'),JSON.stringify({suite,base,counts,notebooks:results},null,2));
  fs.writeFileSync(path.join(root,'summary.md'),markdown);
  if(process.env.GITHUB_STEP_SUMMARY)fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY,markdown);
  log('SUMMARY',counts.PASS+' PASS / '+counts.FAIL+' FAIL / '+counts.SKIP+' SKIP');
  console.log(markdown);
}
async function main() {
  const files=notebooksForSuite();
  log('INFO','Suite='+suite+'; notebooks='+files.length+'; site='+base);
  browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  for(const file of files) {
    log('START',file);
    const r=await recordNotebook(file);
    results.push(r);
    log(r.status,file+'; code='+r.sourceCodeCells+'; executed='+(r.execution?.pass||0)+
      '; errors='+(r.execution?.fail||0)+'; elapsed='+(r.durationSeconds||0)+'s');
    if(r.errors?.length)r.errors.forEach(e=>log('ERROR',file+' '+e.slice(0,2500)));
    summary();
  }
}
main().catch(e=>{
  log('FAIL','Runner exception: '+errorText(e));
  results.push({path:'[runner]',status:'FAIL',sourceCodeCells:0,errors:[errorText(e)]});
}).finally(async()=>{
  if(browser)await browser.close().catch(()=>{});
  summary();
  process.exitCode=0;
});
