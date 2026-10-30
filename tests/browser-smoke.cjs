const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const {chromium} = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8765;
const server = spawn('python', ['-m','http.server',String(port),'--bind','127.0.0.1','--directory','dist'], {stdio:'ignore'});

(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless:true, args:['--no-sandbox']});
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/lab/`,{waitUntil:'domcontentloaded',timeout:45000});
    console.log('JupyterLite page loaded:', await page.title());
    const outcome = await page.evaluate(async (port) => {
      const src=`http://127.0.0.1:${port}/vendor/tf.min.js`;
      const code=`
        self.onmessage = async ({data}) => {
          try {
            const {loadPyodide} = await import('https://cdn.jsdelivr.net/pyodide/v314.0.4/full/pyodide.mjs');
            const py = await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v314.0.4/full/'});
            await py.loadPackage(['numpy','pandas','scipy','matplotlib','scikit-learn']);
            const script = await (await fetch(data.bridge)).text();
            py.FS.writeFile('/home/pyodide/tfjs_bridge.py', script);
            const output = await py.runPythonAsync(
                "import numpy as np, scipy, pandas, matplotlib, sklearn\\n"
              + "from sklearn.linear_model import LinearRegression\\n"
              + "reg=LinearRegression().fit([[1],[2],[3]],[3,5,7])\\n"
              + "assert abs(reg.predict([[4]])[0]-9)<1e-3\\n"
              + "import tfjs_bridge\\n"
              + "await tfjs_bridge.load_tfjs('"+data.tfjs+"')\\n"
              + "model=tfjs_bridge.Sequential([tfjs_bridge.Dense(4,'relu',input_shape=(2,)),tfjs_bridge.Dense(1,'sigmoid')])\\n"
              + "model.compile(loss='binaryCrossentropy',learning_rate=0.05)\\n"
              + "X=np.asarray([[0,0],[0,1],[1,0],[1,1]]*5,dtype='float32')\\n"
              + "y=np.asarray([[0],[1],[1],[1]]*5,dtype='float32')\\n"
              + "h=await model.fit(X,y,epochs=12,batch_size=8)\\n"
              + "pred=model.predict(X)\\n"
              + "assert pred.shape == y.shape\\n"
              + "print('Pyodide WASM scientific imports PASS; TensorFlow.js worker training PASS')\\n"
              + "tfjs_bridge.backend()"
            );
            self.postMessage({ok:true,backend:output});
          } catch (err) {self.postMessage({ok:false,error:String(err),stack:err.stack})}
        }`;
      return await new Promise((resolve,reject) => {
        const blob=URL.createObjectURL(new Blob([code],{type:'text/javascript'}));
        const worker=new Worker(blob,{type:'module'});
        const timer=setTimeout(()=>{worker.terminate();reject(Error('Pyodide/TF.js smoke test timed out'));},150000);
        worker.onmessage=e=>{clearTimeout(timer);worker.terminate();URL.revokeObjectURL(blob);resolve(e.data)};
        worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message||'Worker error'))};
        worker.postMessage({bridge:`http://127.0.0.1:${port}/vendor/tfjs_bridge.py`,tfjs:src});
      });
    },port);
    console.log(JSON.stringify(outcome));
    assert.equal(outcome.ok,true,outcome.error+'\n'+outcome.stack);
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
