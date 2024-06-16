import { RefObject, useEffect, useRef, useState } from 'react'
import './App.css'
import * as handpose from '@tensorflow-models/handpose';
import '@tensorflow/tfjs-backend-webgl'

const useVideo = (videoRef: RefObject<HTMLVideoElement>) => {
  useEffect(() => {
    const setupCamera = async () => {
      if (!videoRef.current) return;
      if (navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true
        });
        videoRef.current.srcObject = stream;
      }
    };

    setupCamera();
  }, [videoRef]);
}

const useHandpose = (videoRef: RefObject<HTMLVideoElement>, canvasRef: RefObject<HTMLCanvasElement>) => {
  const [model, setModel] = useState<handpose.HandPose>();
  useEffect(() => {
    const loadHandpose = async () => {
      const net = await handpose.load();
      setModel(net);
    };

    loadHandpose();
  }, [])

  useEffect(() => {
    const detect = async () => {
      if (!model) return;
      if (!videoRef.current) return;
      const predictions = await model.estimateHands(videoRef.current, true);

      if (!canvasRef.current) return;
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, 640, 480);

      for (let i = 0; i < predictions.length; i++) {
        const keypoints = predictions[i].landmarks;
        for (let j = 0; j < keypoints.length; j++) {
          const [x, y] = keypoints[j];
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 3 * Math.PI);
          ctx.fillStyle = 'aqua';
          ctx.fill();
        }
      }
    }

    const interval = setInterval(detect, 100); // 0.1秒ごとに検出
    return () => clearInterval(interval);
  }, [model, videoRef, canvasRef]);
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useVideo(videoRef);
  useHandpose(videoRef, canvasRef);

  return (
    <>
      <div style={{
        position: 'relative',
        width: 640,
        height: 480,
      }}>
        <video style={{
          position: 'absolute',
          top: 0,
          left: 0,
          transform: 'scaleX(-1)',
        }} ref={videoRef} width="640" height="480" autoPlay />
        <canvas style={{
          position: 'absolute',
          top: 0,
          left: 0,
        }} ref={canvasRef} width="640" height="480" />
      </div>
    </>
  )
}

export default App
