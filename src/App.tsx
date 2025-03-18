import { RefObject, useEffect, useRef, useState } from "react";
import "./App.css";
import * as handpose from "@tensorflow-models/handpose";
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
	const [model, setModel] = useState<handpose.HandPose>();

	useEffect(() => {
		const loadHandpose = async () => {
			const net = await handpose.load();
			setModel(net);
		};

		setIsLoading(true);
		loadHandpose().finally(() => {
			setIsLoading(false)
		});
	}, []);

	useEffect(() => {
		const detect = async () => {
			if (!model) return;
			if (!videoRef.current) return;
			const predictions = await model.estimateHands(videoRef.current, true);

			if (!canvasRef.current) return;
			const ctx = canvasRef.current.getContext("2d");
			if (!ctx) return;

			ctx.clearRect(0, 0, 640, 480);

			for (let i = 0; i < predictions.length; i++) {
				const keypoints = predictions[i].landmarks;
				for (let j = 0; j < keypoints.length; j++) {
					const [x, y] = keypoints[j];
					const mirroredX = 640 - x;
					ctx.beginPath();
					ctx.arc(mirroredX, y, 5, 0, 3 * Math.PI);
					ctx.fillStyle = "aqua";
					ctx.fill();
				}
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

			const faces = await model.estimateFaces(videoRef.current);
			
			const ctx = canvasRef.current.getContext("2d");
			if (!ctx) return;

			ctx.clearRect(0, 0, 640, 480);

			for (let i = 0; i < faces.length; i++) {
				const face = faces[i];
				const keypoints = face.keypoints;
				
				for (let j = 0; j < keypoints.length; j++) {
					const { x, y } = keypoints[j];
					const mirroredX = 640 - x;
					ctx.beginPath();
					ctx.arc(mirroredX, y, 2, 0, 3 * Math.PI);
					ctx.fillStyle = "red";
					ctx.fill();
				}
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
						transform: "scaleX(-1)",
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
