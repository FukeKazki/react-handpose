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
	const [activeTab, setActiveTab] = useState<TabType>("face");
	
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
		<div className="app-container" style={{
			width: "100vw",
			height: "100vh",
			margin: 0,
			padding: 0,
			overflow: "hidden",
			position: "relative",
			fontFamily: "Arial, sans-serif",
			backgroundColor: "#000",
		}}>
			{/* ビデオとキャンバスを全画面表示 */}
			<div
				style={{
					position: "absolute",
					top: 0,
					left: 0,
					width: "100%",
					height: "100%",
					zIndex: 1,
				}}
			>
				<video
					style={{
						position: "absolute",
						top: 0,
						left: 0,
						width: "100%",
						height: "100%",
						objectFit: "cover",
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
						width: "100%",
						height: "100%",
						objectFit: "cover",
					}}
					ref={canvasRef}
					width="640"
					height="480"
				/>
			</div>

			{/* フローティングタイトル */}
			<div style={{
				position: "absolute",
				top: "20px",
				left: "20px",
				zIndex: 10,
				color: "white",
				textShadow: "0 2px 4px rgba(0,0,0,0.5)",
			}}>
				<h1 style={{
					margin: "0 0 4px 0",
					fontSize: "24px",
					fontWeight: "bold",
				}}>AI姿勢検出デモ</h1>
				<p style={{
					margin: 0,
					fontSize: "14px",
					opacity: 0.8,
				}}>TensorFlow.jsを使った手と顔の検出</p>
			</div>

			{/* フローティングタブ切り替え */}
			<div style={{ 
				position: "absolute",
				top: "20px",
				right: "20px",
				zIndex: 10,
				display: "flex",
				gap: "12px",
			}}>
				<button 
					onClick={() => setActiveTab("hand")}
					style={{ 
						padding: "8px 16px", 
						backgroundColor: activeTab === "hand" ? "rgba(52, 152, 219, 0.9)" : "rgba(0, 0, 0, 0.6)",
						color: "white",
						border: "none",
						borderRadius: "30px",
						cursor: "pointer",
						backdropFilter: "blur(4px)",
						boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
						display: "flex",
						alignItems: "center",
						fontWeight: activeTab === "hand" ? "bold" : "normal",
						transition: "all 0.3s ease",
					}}
				>
					<span style={{ marginRight: "6px" }}>✋</span> 手の検出
				</button>
				<button 
					onClick={() => setActiveTab("face")}
					style={{ 
						padding: "8px 16px", 
						backgroundColor: activeTab === "face" ? "rgba(52, 152, 219, 0.9)" : "rgba(0, 0, 0, 0.6)",
						color: "white",
						border: "none",
						borderRadius: "30px",
						cursor: "pointer",
						backdropFilter: "blur(4px)",
						boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
						display: "flex",
						alignItems: "center",
						fontWeight: activeTab === "face" ? "bold" : "normal",
						transition: "all 0.3s ease",
					}}
				>
					<span style={{ marginRight: "6px" }}>😊</span> 顔の検出
				</button>
			</div>

			{/* フローティングヒント */}
			<div style={{ 
				position: "absolute",
				bottom: "20px",
				left: "50%",
				transform: "translateX(-50%)",
				zIndex: 10,
				padding: "10px 20px", 
				backgroundColor: "rgba(0, 0, 0, 0.6)",
				color: "white",
				borderRadius: "30px",
				backdropFilter: "blur(4px)",
				boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
				maxWidth: "80%",
				textAlign: "center",
			}}>
				<p style={{ margin: "0" }}>
					<strong>ヒント:</strong> {activeTab === "hand" 
						? "両手を画面内に表示すると、関節と骨格が検出されます。" 
						: "顔を画面内に表示すると、顔のランドマークが検出されます。"}
				</p>
			</div>

			{/* カメラ許可通知 */}
			{!isAllowed && (
				<div style={{
					position: "absolute",
					top: "50%",
					left: "50%",
					transform: "translate(-50%, -50%)",
					zIndex: 20,
					backgroundColor: "rgba(220, 53, 69, 0.9)",
					color: "white",
					padding: "20px 30px",
					borderRadius: "8px",
					backdropFilter: "blur(10px)",
					boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
					textAlign: "center",
					minWidth: "300px",
				}}>
					<div style={{ fontSize: "32px", marginBottom: "10px" }}>📷</div>
					<p style={{ margin: "0", fontWeight: "bold", fontSize: "16px" }}>カメラへのアクセスが必要です</p>
					<p style={{ margin: "8px 0 0 0", fontSize: "14px" }}>このアプリはカメラを使用して手と顔を検出します</p>
				</div>
			)}
			
			{/* ローディング表示 */}
			{isLoading && (
				<div style={{
					position: "absolute",
					top: "50%",
					left: "50%",
					transform: "translate(-50%, -50%)",
					zIndex: 20,
					backgroundColor: "rgba(0, 0, 0, 0.7)",
					color: "white",
					padding: "20px 30px",
					borderRadius: "8px",
					backdropFilter: "blur(10px)",
					boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
					textAlign: "center",
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					justifyContent: "center",
					minWidth: "200px",
				}}>
					<div style={{
						border: "3px solid rgba(255, 255, 255, 0.1)",
						borderTop: "3px solid #fff",
						borderRadius: "50%",
						width: "30px",
						height: "30px",
						animation: "spin 1s linear infinite",
						marginBottom: "12px",
					}} />
					<p style={{ margin: "0", fontWeight: "bold" }}>モデルを読み込み中...</p>
				</div>
			)}
			
			<style>{`
				@keyframes spin {
					0% { transform: rotate(0deg); }
					100% { transform: rotate(360deg); }
				}
				body {
					margin: 0;
					padding: 0;
					overflow: hidden;
				}
				#root {
					max-width: none !important;
					width: 100vw;
					height: 100vh;
					margin: 0;
					padding: 0;
					overflow: hidden;
				}
			`}</style>
		</div>
	);
}

export default App;
