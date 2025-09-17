// src/ml/detector.ts
import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';

let session: ort.InferenceSession | null = null;

const ASSET_NAME = 'models/model.onnx'; // en android/app/src/main/assets/models/model.onnx
const LOCAL_NAME = 'model.onnx';

async function ensureLocalModelFile(): Promise<string> {
  // Ruta destino en almacenamiento interno de la app
  const destPath = `${RNFS.DocumentDirectoryPath}/${LOCAL_NAME}`; // p.ej. /data/user/0/<pkg>/files/model.onnx

  // Si ya existe y pesa > 0, lo reutilizamos
  try {
    const s = await RNFS.stat(destPath);
    if (s.isFile() && s.size > 0) return destPath;
  } catch {}

  // Copiar desde assets -> destino
  // Android: usa copyFileAssets (lee de android/app/src/main/assets)
  await RNFS.copyFileAssets(ASSET_NAME, destPath);

  // Verificación rápida
  const verify = await RNFS.stat(destPath);
  if (!verify.isFile() || verify.size === 0) {
    throw new Error(`Copia de assets fallida: ${destPath}`);
  }
  return destPath;
}

export async function loadModel() {
  if (session) return session;
  if (Platform.OS !== 'android') throw new Error('Solo Android soportado en esta ruta');

  // 1) Garantiza archivo local real
  const localPath = await ensureLocalModelFile(); // ej: /data/.../model.onnx
  const fileUri = `file://${localPath}`;

  // 2) Crea la sesión (no fuerces executionProviders)
  session = await ort.InferenceSession.create(fileUri);

  console.log('✅ Modelo cargado desde archivo local:', fileUri);
  return session;
}
