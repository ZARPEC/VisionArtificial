// App.tsx
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, Pressable, StyleSheet, Alert, Vibration, LayoutChangeEvent } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import type { Camera as CameraRef } from 'react-native-vision-camera';
import type { DetOut } from './android/app/src/ml/detector';
import Tts from 'react-native-tts';
import { Buffer } from 'buffer';
(global as any).Buffer = (global as any).Buffer || Buffer;

// ML utils
import { loadModelFromLocal, inferOnJpegFileResized } from './android/app/src/ml/detector';


// Tipo simple para las detecciones (coincide con lo que devuelve inferOnJpegFileResized)
type Box = DetOut;

const INPUT_SIZE = 640;

export default function App() {
  const device = useCameraDevice('back');
  const camRef = useRef<CameraRef>(null);

  const [running, setRunning] = useState(false);
  const [hasPerms, setHasPerms] = useState<null | boolean>(null);
  const [modelStatus, setModelStatus] = useState('⏳ Cargando modelo…');
  const [inferText, setInferText] = useState('—');

  // 👇 para overlay
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [previewW, setPreviewW] = useState(0);
  const [previewH, setPreviewH] = useState(0);

  // Pedir permisos
  useEffect(() => {
    (async () => {
      const cam = await Camera.requestCameraPermission();
      const mic = await Camera.requestMicrophonePermission();
      setHasPerms(cam === 'granted' && mic === 'granted');
    })();
    Tts.setDefaultLanguage('es-MX').catch(() => {});
  }, []);

  // Cargar modelo ONNX
  useEffect(() => {
    (async () => {
      try {
        await loadModelFromLocal();
        setModelStatus('✅ ONNX listo');
      } catch (e: any) {
        setModelStatus('❌ ONNX: ' + (e?.message ?? String(e)));
      }
    })();
  }, []);

  // Voz al iniciar/detener
  useEffect(() => {
    Tts.stop();
    Tts.speak(running ? 'Exploración iniciada' : 'Exploración detenida');
  }, [running]);

  // Feedback simple cada 3s
 
  const toggle = () => {
    if (!hasPerms) {
      Alert.alert('Permisos', 'Habilita cámara y micrófono para continuar.');
      return;
    }
    setRunning(v => !v);
  };

  // Medir tamaño real del contenedor del preview (para escalar cajas)
  const onPreviewLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setPreviewW(width);
    setPreviewH(height);
  };

  // Captura + detección
  const onCaptureAndDetect = async () => {
    try {
      if (!camRef.current) { Alert.alert('Cámara', 'No disponible'); return; }
      if (!hasPerms) { Alert.alert('Permisos', 'Habilita cámara y micrófono.'); return; }

      setInferText('⏳ Tomando foto…');
      const photo = await camRef.current.takePhoto({ flash: 'off' });

      setInferText('⏳ Preparando…');
      const { dets } = await inferOnJpegFileResized(photo.path, (msg)=>setInferText('⏳ '+msg));

      setBoxes(dets.map(d => ({ ...d, zone: d.zone as 'L'|'C'|'R' })));


      if (!dets.length) {
        setInferText('✅ Sin detecciones ≥ umbral');
        Tts.stop(); Tts.speak('Sin detecciones');
        return;
      }

      const topDet = dets.slice(0,5);
      const lines = topDet.map((d,i)=>`#${i+1} ${d.name ?? `cls${d.cls}`} (${d.zone}) score=${d.score.toFixed(2)}`).join('\n');
      setInferText(`✅ ${dets.length} detección(es)\n${lines}`);

      // TTS resumido por zonas
      const pick = (z:'L'|'C'|'R') => topDet.find(d=>d.zone===z);
      const left   = pick('L')?.name;
      const center = pick('C')?.name;
      const right  = pick('R')?.name;
      let msg = '';
      if (left)   msg += `${left} a la izquierda. `;
      if (center) msg += `${center} al frente. `;
      if (right)  msg += `${right} a la derecha. `;
      if (!msg) msg = 'Objetos detectados.';

      Tts.stop(); Tts.speak(msg.trim());
      Vibration.vibrate(60);
    } catch (e:any) {
      setInferText('❌ Error: ' + (e?.message ?? String(e)));
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Asistente de Visión</Text>
      <Text style={styles.modelStatus}>{modelStatus}</Text>

      <View style={styles.preview} onLayout={onPreviewLayout}>
        {device && hasPerms ? (
          <>
            <Camera
              ref={camRef}
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={running}
              photo={true}
            />
            {/* 👇 Overlay */}
            <BoxesOverlay boxes={boxes} width={previewW} height={previewH} />
          </>
        ) : (
          <Text style={styles.placeholder}>
            {hasPerms === false ? 'Permisos rechazados' : 'Comprobando permisos...'}
          </Text>
        )}
      </View>

      <Pressable style={[styles.button, running && styles.buttonStop]} onPress={toggle}>
        <Text style={styles.buttonText}>{running ? 'Detener' : 'Explorar'}</Text>
      </Pressable>

      <Pressable style={[styles.button, { marginTop: 8 }]} onPress={onCaptureAndDetect}>
        <Text style={styles.buttonText}>Tomar foto y detectar</Text>
      </Pressable>

      <Text style={styles.status}>Estado: {running ? 'Analizando…' : 'Listo'}</Text>
      <Text style={styles.infer}>{inferText}</Text>
    </View>
  );
}

/** ================= Overlay de cajas ================= **/
function BoxesOverlay({ boxes, width, height }: { boxes: Box[]; width: number; height: number }) {
  if (!width || !height || boxes.length === 0) return null;

  // Escala y offsets para simular letterbox a 640x640 dentro del preview real
  const scale = Math.min(width / INPUT_SIZE, height / INPUT_SIZE);
  const padX = (width  - INPUT_SIZE * scale) / 2;
  const padY = (height - INPUT_SIZE * scale) / 2;

  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {boxes.map((b, idx) => {
        const x = b.x1 * scale + padX;
        const y = b.y1 * scale + padY;
        const w = (b.x2 - b.x1) * scale;
        const h = (b.y2 - b.y1) * scale;
        return (
          <View key={idx} style={[styles.box, { left: x, top: y, width: w, height: h }]}>
            <Text style={styles.boxLabel}>
              {(b.name ?? `cls${b.cls}`)} {(b.zone ?? '')} {b.score.toFixed(2)}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000', padding: 16, gap: 16 },
  title: { color: '#fff', fontSize: 22, textAlign: 'center', marginTop: 8 },
  modelStatus: { color: '#0f0', fontSize: 16, textAlign: 'center', marginBottom: 4 },
  preview: {
    flex: 1,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#111',
    borderWidth: 1,
    borderColor: '#222',
  },
  placeholder: { color: '#aaa', textAlign: 'center', marginTop: 20 },
  button: { backgroundColor: '#1e90ff', paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  buttonStop: { backgroundColor: '#ff4d4d' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  status: { color: '#ccc', textAlign: 'center', marginBottom: 6 },
  infer: { color: '#9cf', fontSize: 13, textAlign: 'center', marginTop: 8, includeFontPadding: false },

  // Overlay styles
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00e5ff',
    borderRadius: 6,
  },
  boxLabel: {
    position: 'absolute',
    left: 0,
    top: -22,
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: 'rgba(0,0,0,0.7)',
    color: '#fff',
    fontSize: 12,
    borderTopLeftRadius: 6,
    borderTopRightRadius: 6,
  },
});
