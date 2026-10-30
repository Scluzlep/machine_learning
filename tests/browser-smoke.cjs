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
            const {loadPyodide} = await import(data.pyodide);
            const py = await loadPyodide({indexURL:data.pyodideBase});
            await py.loadPackage(['numpy','pandas','scipy','matplotlib','scikit-learn','micropip']);
            await py.runPythonAsync("import micropip; await micropip.install(['ipywidgets', 'ipympl']); import ipywidgets, ipympl; slider = ipywidgets.IntSlider(value=4); assert slider.value == 4");
            await py.loadPackage(['h5py', 'pillow']);
            await py.runPythonAsync(
                "import h5py\\n"
              + "from PIL import Image\\n"
              + "import numpy as np\\n"
              + "with h5py.File('/tmp/ml-wheel-probe.h5', 'w') as f:\\n"
              + "    f['values'] = np.array([2, 3, 5])\\n"
              + "with h5py.File('/tmp/ml-wheel-probe.h5', 'r') as f:\\n"
              + "    assert f['values'][:].tolist() == [2, 3, 5]\\n"
              + "assert Image.new('RGB', (3, 2)).size == (3, 2)\\n"
            );
            const script = await (await fetch(data.bridge)).text();
            py.FS.writeFile('/home/pyodide/tfjs_bridge.py', script);
            py.FS.mkdirTree('/home/pyodide/browser_ml');
            py.FS.writeFile('/home/pyodide/browser_ml/__init__.py', '');
            py.FS.writeFile('/home/pyodide/browser_ml/tfjs_bridge.py', script);
            const compatText = await (await fetch(data.compat)).text();
            py.FS.writeFile('/home/pyodide/browser_ml/tfjs_compat.py', compatText);
            const output = await py.runPythonAsync(
                "import numpy as np, scipy, pandas, matplotlib, sklearn\\n"
              + "from sklearn.linear_model import LinearRegression\\n"
              + "reg=LinearRegression().fit([[1],[2],[3]],[3,5,7])\\n"
              + "assert abs(reg.predict([[4]])[0]-9)<1e-3\\n"
              + "import tfjs_bridge\\n"
              + "await tfjs_bridge.load_tfjs('"+data.tfjs+"', backend='cpu')\\n"
              + "assert tfjs_bridge.backend() == 'cpu'\\n"
              + "await tfjs_bridge.load_tfjs('"+data.tfjs+"', backend='auto')\\n"
              + "assert tfjs_bridge.backend() in ('webgpu', 'webgl', 'cpu')\\n"
              + "model=tfjs_bridge.Sequential([tfjs_bridge.Dense(4,'relu',input_shape=(2,)),tfjs_bridge.Dense(1,'sigmoid')])\\n"
              + "model.compile(loss='binaryCrossentropy',learning_rate=0.05)\\n"
              + "X=np.asarray([[0,0],[0,1],[1,0],[1,1]]*5,dtype='float32')\\n"
              + "y=np.asarray([[0],[1],[1],[1]]*5,dtype='float32')\\n"
              + "h=await model.fit(X,y,epochs=12,batch_size=8)\\n"
              + "pred=model.predict(X)\\n"
              + "assert pred.shape == y.shape\\n"
              + "print('Pyodide WASM scientific imports PASS; TensorFlow.js worker training PASS')\\n"
              + "str(tfjs_bridge.backend_info())"
            );
            globalThis.eval(await (await fetch(data.losses)).text());
            await py.runPythonAsync(
                "from browser_ml.tfjs_compat import install\\n"
              + "tf = await install(backend='cpu')\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense\\n"
              + "from tensorflow.keras.activations import sigmoid\\n"
              + "import numpy as np\\n"
              + "norm = tf.keras.layers.Normalization(axis=-1)\\n"
              + "norm.adapt(np.asarray([[1,2],[3,4]], dtype='float32'))\\n"
              + "assert np.isfinite(norm(np.asarray([[2,3]]))).all()\\n"
              + "assert abs(float(sigmoid(0).numpy())-0.5)<1e-5\\n"
              + "model2 = Sequential([tf.keras.Input(shape=(2,)), Dense(3,'relu',name='layer1'), Dense(1,'sigmoid',name='layer2')])\\n"
              + "weights = model2.get_layer('layer1').get_weights()\\n"
              + "assert len(weights)==2 and weights[0].shape==(2,3)\\n"
              + "model2.get_layer('layer1').set_weights(weights)\\n"
              + "model2.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))\\n"
              + "hist2 = await model2.fit([[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[1]],epochs=2)\\n"
              + "assert model2.predict([[0,0]]).shape==(1,1)\\n"
              + "model2.dispose()\\n"
            );
            await py.runPythonAsync(
                "import numpy as np\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense\\n"
              + "import tensorflow as tf\\n"
              + "m3=Sequential([Dense(8,'relu'), Dense(4,'linear')])\\n"
              + "m3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(0.03))\\n"
              + "xc=np.asarray([[0,0],[0,1],[1,0],[1,1]] * 12,dtype='float32')\\n"
              + "yc=np.asarray([0,1,2,3]*12,dtype='int32')\\n"
              + "hist3=await m3.fit(xc,yc,epochs=18,batch_size=12)\\n"
              + "assert 'loss' in hist3.history and len(hist3.history['loss'])==18\\n"
              + "assert np.isfinite(hist3.history['loss']).all()\\n"
              + "assert hist3.history['loss'][-1] < hist3.history['loss'][0]\\n"
              + "assert m3.predict(xc[:4]).shape == (4,4)\\n"
              + "assert m3.layers[0].get_weights()[0].shape == (2,8)\\n"
              + "m3.dispose()\\n"
            );
            await py.runPythonAsync(
                "import numpy as np\\n"
              + "import tensorflow as tf\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense\\n"
              + "from tensorflow.keras.activations import relu,linear\\n"
              + "from tensorflow.keras.losses import SparseCategoricalCrossentropy\\n"
              + "from tensorflow.keras.optimizers import Adam\\n"
              + "from tensorflow.keras.regularizers import l2\\n"
              + "m4=Sequential([Dense(4,activation='relu',kernel_regularizer=l2(0.1),name='L1'),Dense(3,activation='linear',name='L2')])\\n"
              + "m4.compile(loss=SparseCategoricalCrossentropy(from_logits=True),optimizer=Adam(lr=0.01))\\n"
              + "m4.build((None,2))\\n"
              + "assert m4.input.shape.as_list()==[None,2]\\n"
              + "assert m4.layers[0].output.shape.as_list()==[None,4]\\n"
              + "assert m4.layers[0].activation is relu and m4.layers[1].activation is linear\\n"
              + "assert type(m4.layers[0].kernel_regularizer) is l2\\n"
              + "assert m4.loss.get_config()['from_logits'] and isinstance(m4.optimizer,Adam)\\n"
              + "assert np.isclose(m4.optimizer.learning_rate.numpy(),0.01)\\n"
              + "xs=np.asarray([[0,0],[1,0],[0,1],[1,1]]*2,dtype='float32')\\n"
              + "ys=np.asarray([0,1,2,1]*2,dtype='int32')\\n"
              + "h4=await m4.fit(xs,ys,epochs=2)\\n"
              + "assert np.isfinite(h4.history['loss']).all()\\n"
              + "assert (await m4.predict_async(xs)).shape==(8,3)\\n"
              + "assert len(await m4.get_weights_async())==4\\n"
              + "m4.dispose()\\n"
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
        worker.postMessage({pyodide:`http://127.0.0.1:${port}/pyodide/pyodide.mjs`,pyodideBase:`http://127.0.0.1:${port}/pyodide/`,bridge:`http://127.0.0.1:${port}/vendor/tfjs_bridge.py`,compat:`http://127.0.0.1:${port}/vendor/tfjs_compat.py`,losses:`http://127.0.0.1:${port}/vendor/course-losses.js`,tfjs:src});
      });
    },port);
    console.log('Browser scientific/TensorFlow.js backend test:', JSON.stringify(outcome));
    assert.equal(outcome.ok,true,outcome.error+'\n'+outcome.stack);
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
