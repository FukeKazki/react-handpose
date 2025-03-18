import { RefObject, useEffect, useRef, useState } from "react";
import "./App.css";
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";

type TabType = "hand" | "face";

const useVideo = (videoRef: RefObject<HTMLVideoElement>) => {
	const [isAllowed, setIsAllowed] = useState(false);
	useEffect(() => {
		const setupCamera = async () => {
			if (!videoRef.current) return;
			if (navigator.mediaDevices.getUserMedia) {
				const stream = await navigator.mediaDevices.getUserMedia({
					video: true,
				});
				videoRef.current.srcObject = stream;
			}
		};

		setupCamera().then(() => {
			setIsAllowed(true);
		});
	}, [videoRef]);

	return {
		isAllowed,
	}
};

const useHandpose = (
	videoRef: RefObject<HTMLVideoElement>,
	canvasRef: RefObject<HTMLCanvasElement>,
) => {
	const [isLoading, setIsLoading] = useState(false);
	const [model, setModel] = useState<handPoseDetection.HandDetector>();

	useEffect(() => {
		const loadHandpose = async () => {
			try {
				// MediaPipeHandsモデルを使用
				const model = handPoseDetection.SupportedModels.MediaPipeHands;
				const detectorConfig = {
					runtime: 'tfjs' as const,
					modelType: 'full' as const,
					maxHands: 2  // 最大2つの手を検出
				};
				const detector = await handPoseDetection.createDetector(
					model,
					detectorConfig
				);
				
				setModel(detector);
			} catch (error) {
				console.error("手の検出モデルの読み込みエラー:", error);
			}
		};

		setIsLoading(true);
		loadHandpose().finally(() => {
			setIsLoading(false)
		});
	}, []);

	useEffect(() => {
		// 接続するランドマークのペア（骨格を描画するため）
		const connections = [
			// 親指
			[0, 1], [1, 2], [2, 3], [3, 4],
			// 人差し指
			[0, 5], [5, 6], [6, 7], [7, 8],
			// 中指
			[0, 9], [9, 10], [10, 11], [11, 12],
			// 薬指
			[0, 13], [13, 14], [14, 15], [15, 16],
			// 小指
			[0, 17], [17, 18], [18, 19], [19, 20],
			// 手のひら
			[0, 5], [5, 9], [9, 13], [13, 17], [0, 17]
		];

		const detect = async () => {
			if (!model) return;
			if (!videoRef.current) return;
			
			try {
				const hands = await model.estimateHands(videoRef.current);

				if (!canvasRef.current) return;
				const ctx = canvasRef.current.getContext("2d");
				if (!ctx) return;

				ctx.clearRect(0, 0, 640, 480);

				// 検出された各手について処理
				for (let i = 0; i < hands.length; i++) {
					const hand = hands[i];
					const keypoints = hand.keypoints;
					
					// 各ランドマークを描画
					for (let j = 0; j < keypoints.length; j++) {
						const { x, y } = keypoints[j];
						ctx.beginPath();
						ctx.arc(x, y, 5, 0, 3 * Math.PI);
						ctx.fillStyle = "aqua";
						ctx.fill();
					}
					
					// 骨格（指の接続）を描画
					for (const [start, end] of connections) {
						const startPoint = keypoints[start];
						const endPoint = keypoints[end];
						
						if (startPoint && endPoint) {
							ctx.beginPath();
							ctx.moveTo(startPoint.x, startPoint.y);
							ctx.lineTo(endPoint.x, endPoint.y);
							ctx.strokeStyle = "yellow";
							ctx.lineWidth = 2;
							ctx.stroke();
						}
					}
					
					// 手の種類（左/右）を表示
					const handedness = hand.handedness; // 'Left' または 'Right'
					// ビデオ反転を削除したので、そのまま表示
					const handText = `${handedness} Hand`;
					ctx.font = "16px Arial";
					ctx.fillStyle = "white";
					
					// 手首の位置の近くにテキストを表示
					const wrist = keypoints[0];
					if (wrist) {
						ctx.fillText(handText, wrist.x - 20, wrist.y - 10);
					}
				}
			} catch (error) {
				console.error("手の検出エラー:", error);
			}
		};

		const interval = setInterval(detect, 100); // 0.1秒ごとに検出
		return () => clearInterval(interval);
	}, [model, videoRef, canvasRef]);

	return {
		isLoading,
	}
};

const useFaceDetection = (
	videoRef: RefObject<HTMLVideoElement>,
	canvasRef: RefObject<HTMLCanvasElement>,
) => {
	const [isLoading, setIsLoading] = useState(false);
	const [model, setModel] = useState<faceLandmarksDetection.FaceLandmarksDetector>();

	useEffect(() => {
		const loadFaceDetection = async () => {
			try {
				const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
				const detectorConfig = {
					runtime: 'tfjs',
					refineLandmarks: false,
					maxFaces: 1
				} as const;
				const detector = await faceLandmarksDetection.createDetector(
					model,
					detectorConfig
				);
				setModel(detector);
			} catch (error) {
				console.error("顔検出モデルの読み込みエラー:", error);
			}
		};

		setIsLoading(true);
		loadFaceDetection().finally(() => {
			setIsLoading(false);
		});
	}, []);

	useEffect(() => {
		const detect = async () => {
			if (!model) return;
			if (!videoRef.current) return;
			if (!canvasRef.current) return;

			try {
				const faces = await model.estimateFaces(videoRef.current);
				
				const ctx = canvasRef.current.getContext("2d");
				if (!ctx) return;

				ctx.clearRect(0, 0, 640, 480);

				for (let i = 0; i < faces.length; i++) {
					const face = faces[i];
					const keypoints = face.keypoints;
					
					for (let j = 0; j < keypoints.length; j++) {
						const { x, y } = keypoints[j];
						ctx.beginPath();
						ctx.arc(x, y, 2, 0, 3 * Math.PI);
						ctx.fillStyle = "red";
						ctx.fill();
					}
				}
			} catch (error) {
				console.error("顔の検出エラー:", error);
			}
		};

		const interval = setInterval(detect, 100); // 0.1秒ごとに検出
		return () => clearInterval(interval);
	}, [model, videoRef, canvasRef]);

	return {
		isLoading,
	}
};

function App() {
	const videoRef = useRef<HTMLVideoElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const { isAllowed } = useVideo(videoRef);
	const [activeTab, setActiveTab] = useState<TabType>("hand");
	
	const { isLoading: isHandLoading } = useHandpose(
		activeTab === "hand" ? videoRef : { current: null },
		activeTab === "hand" ? canvasRef : { current: null }
	);
	
	const { isLoading: isFaceLoading } = useFaceDetection(
		activeTab === "face" ? videoRef : { current: null },
		activeTab === "face" ? canvasRef : { current: null }
	);

	const isLoading = activeTab === "hand" ? isHandLoading : isFaceLoading;

	return (
		<>
			{isAllowed || <p>Camera Permission Denied</p>}
			{isLoading && <p>Model Loading...</p>}
			
			<div style={{ marginBottom: "16px" }}>
				<button 
					onClick={() => setActiveTab("hand")}
					style={{ 
						padding: "8px 16px", 
						backgroundColor: activeTab === "hand" ? "#3498db" : "#f1f1f1",
						color: activeTab === "hand" ? "white" : "black",
						border: "none",
						borderRadius: "4px",
						marginRight: "8px",
						cursor: "pointer"
					}}
				>
					手の検出
				</button>
				<button 
					onClick={() => setActiveTab("face")}
					style={{ 
						padding: "8px 16px", 
						backgroundColor: activeTab === "face" ? "#3498db" : "#f1f1f1",
						color: activeTab === "face" ? "white" : "black",
						border: "none",
						borderRadius: "4px",
						cursor: "pointer"
					}}
				>
					顔の検出
				</button>
			</div>
			
			<div
				style={{
					position: "relative",
					width: 640,
					height: 480,
				}}
			>
				<video
					style={{
						position: "absolute",
						top: 0,
						left: 0,
					}}
					ref={videoRef}
					width="640"
					height="480"
					autoPlay
				/>
				<canvas
					style={{
						position: "absolute",
						top: 0,
						left: 0,
					}}
					ref={canvasRef}
					width="640"
					height="480"
				/>
			</div>
		</>
	);
}

export default App;
