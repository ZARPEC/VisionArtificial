// src/ml/detector.ts
import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import ImageResizer from 'react-native-image-resizer';
import jpeg from 'jpeg-js';

let session: ort.InferenceSession | null = null;

const ASSET_MODEL = 'models/model.onnx';
const LOCAL_NAME = 'model.onnx';
const INPUT_SIZE = 640;
const NUM_CLASSES = 80;
const SCORE_THR = 0.55;
const IOU_THR = 0.50;

const COCO = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
  "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
  "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
  "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
  "chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
  "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
  "vase","scissors","teddy bear","hair drier","toothbrush"
];

function sigmoid(x:number){ return 1/(1+Math.exp(-x)); }

type Det = { x1:number; y1:number; x2:number; y2:number; score:number; cls:number; name?: string; zone?:'L'|'C'|'R'};

function iou(a: Det, b: Det){
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw*ih;
  const ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
  return ua ? inter/ua : 0;
}
function nms(dets: Det[], thr=IOU_THR){
  dets.sort((a,b)=>b.score-a.score);
  const keep: Det[] = [];
  dets.forEach(d=>{ if(keep.every(k=>iou(k,d)<thr)) keep.push(d); });
  return keep;
}
function scaleBoxes(boxes: Det[], inW:number, inH:number, outW:number, outH:number){
  const r = Math.min(outW/inW, outH/inH);
  const padW = (outW - inW*r)/2;
  const padH = (outH - inH*r)/2;
  return boxes.map(b=>{
    const x1 = Math.max(0, (b.x1 - padW)/r);
    const y1 = Math.max(0, (b.y1 - padH)/r);
    const x2 = Math.min(inW, (b.x2 - padW)/r);
    const y2 = Math.min(inH, (b.y2 - padH)/r);
    return { ...b, x1, y1, x2, y2 };
  });
}
function letterboxResizeRGBA(
  rgba: Uint8Array, srcW:number, srcH:number, dst: Float32Array, size=INPUT_SIZE
){
  const r = Math.min(size/srcW, size/srcH);
  const newW = Math.round(srcW*r), newH = Math.round(srcH*r);
  const dx = Math.floor((size - newW)/2), dy = Math.floor((size - newH)/2);
  const rgb = new Uint8Array(newW*newH*3);
  for(let y=0;y<newH;y++){
    const sy = Math.min(srcH-1, Math.round(y/r));
    for(let x=0;x<newW;x++){
      const sx = Math.min(srcW-1, Math.round(x/r));
      const si = (sy*srcW + sx)*4;
      const di = (y*newW + x)*3;
      rgb[di] = rgba[si]; rgb[di+1] = rgba[si+1]; rgb[di+2] = rgba[si+2];
    }
  }
  for(let y=0;y<newH;y++){
    for(let x=0;x<newW;x++){
      const si = (y*newW + x)*3;
      const ox = x+dx, oy = y+dy;
      const baseR = (0*INPUT_SIZE + oy)*INPUT_SIZE + ox;
      const baseG = (1*INPUT_SIZE + oy)*INPUT_SIZE + ox;
      const baseB = (2*INPUT_SIZE + oy)*INPUT_SIZE + ox;
      dst[baseR] = rgb[si]/255; dst[baseG] = rgb[si+1]/255; dst[baseB] = rgb[si+2]/255;
    }
  }
}

async function ensureLocalModelFile(): Promise<string> {
  const dest = `${RNFS.DocumentDirectoryPath}/${LOCAL_NAME}`;
  try {
    const s = await RNFS.stat(dest);
    if (s.isFile() && s.size>0) return dest;
  } catch {}
  try { await RNFS.mkdir(RNFS.DocumentDirectoryPath); } catch {}
  await RNFS.copyFileAssets(ASSET_MODEL, dest);
  const v = await RNFS.stat(dest);
  if (!v.isFile() || v.size<=0) throw new Error(`Copia de modelo fallida: ${dest}`);
  return dest;
}

export async function loadModelFromLocal(): Promise<ort.InferenceSession> {
  if (session) return session;
  const local = await ensureLocalModelFile();
  const uri = `file://${local}`;
  session = await ort.InferenceSession.create(uri);
  return session;
}

export async function inferOnJpegFileResized(
  photoPath: string,
  progress?: (msg:string)=>void
){
  // 1) Redimensiona de forma nativa (¡rápido!)
  progress?.('Redimensionando foto…');
  const resized = await ImageResizer.createResizedImage(
    photoPath,
    INPUT_SIZE, INPUT_SIZE,   // tamaño máximo por lado
    'JPEG',                   // formato
    75,                       // calidad
    0,                        // rotación
    undefined,                // path destino (auto)
    false,                    // keepMeta
    { mode: 'contain', onlyScaleDown: true } // conserva aspecto (letterbox externo)
  );

  // 2) Decodifica JPEG (ya pequeño) a RGBA
  progress?.('Decodificando…');
  const b64 = await RNFS.readFile(resized.path, 'base64');
  const buf = Buffer.from(b64, 'base64');
  const decoded = jpeg.decode(buf, { useTArray:true });
  const rgba = decoded.data as Uint8Array;
  const w = decoded.width, h = decoded.height;

  // 3) Preprocesa a tensor [1,3,640,640]
  progress?.('Preprocesando…');
  const inputData = new Float32Array(1*3*INPUT_SIZE*INPUT_SIZE);
  letterboxResizeRGBA(rgba, w, h, inputData, INPUT_SIZE);
  const input = new ort.Tensor('float32', inputData, [1,3,INPUT_SIZE,INPUT_SIZE]);

  // 4) Inferencia
  progress?.('Inferencia…');
  if (!session) await loadModelFromLocal();
  const inName = session!.inputNames[0];
  const out = await session!.run({ [inName]: input });

  // 5) Post-proceso (YOLOv8: [1,84,8400])
  progress?.('Post-proceso…');
  const outName = Object.keys(out)[0];
  const output = out[outName];
  const arr = output.data as Float32Array;
  const [_, ch, n] = output.dims;

  const dets: Det[] = [];
  for (let i=0;i<n;i++){
    const cx = arr[0*n + i], cy = arr[1*n + i], ww = arr[2*n + i], hh = arr[3*n + i];
    let bestC=-1, bestS=0;
    for (let c=0;c<NUM_CLASSES;c++){
      const s = sigmoid(arr[(4+c)*n + i]);
      if (s>bestS){ bestS=s; bestC=c; }
    }
    if (bestS < SCORE_THR) continue;
    const x1 = cx - ww/2, y1 = cy - hh/2, x2 = cx + ww/2, y2 = cy + hh/2;
    dets.push({ x1, y1, x2, y2, score: bestS, cls: bestC });
  }

 const pruned = nms(dets, IOU_THR);

 
// etiqueta y zona por posición horizontal (centro de la caja en 640)
const enriched = pruned.map(d => {
  const cx = (d.x1 + d.x2) / 2;
  const zone = cx < INPUT_SIZE/3 ? 'L' : (cx > (2*INPUT_SIZE/3) ? 'R' : 'C');
  return { ...d, name: COCO[d.cls] || `cls${d.cls}`, zone };
});

return { dets: enriched };
}


export type DetOut = {
  x1:number; y1:number; x2:number; y2:number;
  score:number; cls:number;
  name?: string;
  zone?: 'L'|'C'|'R';
};
