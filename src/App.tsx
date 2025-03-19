import { RefObject, useEffect, useRef, useState } from "react";
import "./App.css";
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

// アプリ起動時にTensorFlow.jsのバックエンドを初期化
tf.setBackend('webgl').then(() => {
	console.log('TensorFlow.js backend initialized:', tf.getBackend());
}).catch(err => {
	console.error('Failed to initialize TensorFlow.js backend:', err);
});

type TabType = "hand" | "face" | "pose";

const useVideo = (
	videoRef: RefObject<HTMLVideoElement>,
	videoFile: File | null
) => {
	const [isAllowed, setIsAllowed] = useState(false);
	const [isVideoReady, setIsVideoReady] = useState(false);

	// ファイルが選択された場合の処理
	useEffect(() => {
		if (videoFile && videoRef.current) {
			console.log("動画ファイルが選択されました:", videoFile.name);
			
			// カメラのストリームを停止
			const mediaStream = videoRef.current.srcObject as MediaStream;
			if (mediaStream) {
				mediaStream.getTracks().forEach(track => {
					track.stop();
					console.log("カメラトラック停止:", track.kind);
				});
			}

			// 動画の準備状態をリセット
			setIsVideoReady(false);

			try {
				// ファイルからの動画を設定
				const fileURL = URL.createObjectURL(videoFile);
				console.log("動画ファイルURL:", fileURL);
				
				// video要素をリセット
				videoRef.current.pause();
				videoRef.current.removeAttribute('srcObject');
				videoRef.current.srcObject = null;
				videoRef.current.src = fileURL;
				videoRef.current.muted = false;
				videoRef.current.crossOrigin = "anonymous";
				videoRef.current.load();
				
				console.log("動画要素設定完了");
				
				// メタデータが読み込まれたら準備完了とマーク
				videoRef.current.onloadedmetadata = () => {
					console.log("動画メタデータ読み込み完了:", videoRef.current?.videoWidth, "x", videoRef.current?.videoHeight);
					
					// メタデータが読み込まれた後、実際の再生開始
					videoRef.current?.play().then(() => {
						console.log("動画再生開始");
						setIsVideoReady(true);
					}).catch(err => {
						console.error("動画再生エラー:", err);
					});
				};
				
				// 再生終了時にループ再生
				videoRef.current.onended = () => {
					console.log("動画再生終了、ループします");
					if (videoRef.current) {
						videoRef.current.currentTime = 0;
						videoRef.current.play().catch(err => {
							console.error("動画ループ再生エラー:", err);
						});
					}
				};
				
				// エラーハンドリング
				videoRef.current.onerror = (e) => {
					console.error("動画読み込みエラー:", e);
				};
				
				setIsAllowed(true);
			} catch (error) {
				console.error("動画設定エラー:", error);
			}
			
			// クリーンアップ
			return () => {
				if (videoRef.current) {
					const oldSrc = videoRef.current.src;
					videoRef.current.onloadedmetadata = null;
					videoRef.current.onended = null;
					videoRef.current.onerror = null;
					videoRef.current.pause();
					videoRef.current.src = "";
					videoRef.current.load();
					if (oldSrc) {
						URL.revokeObjectURL(oldSrc);
					}
					console.log("動画リソース解放");
				}
				setIsVideoReady(false);
			};
		} else if (!videoFile) {
			// ファイルがない場合はカメラを使用
			const setupCamera = async () => {
				if (!videoRef.current) return;
				try {
					if (navigator.mediaDevices.getUserMedia) {
						const stream = await navigator.mediaDevices.getUserMedia({
							video: true,
						});
						videoRef.current.srcObject = stream;
						videoRef.current.muted = true; // カメラ映像はミュート
						
						// カメラのメタデータが読み込まれたら準備完了とマーク
						videoRef.current.onloadedmetadata = () => {
							console.log("カメラ映像メタデータ読み込み完了");
							setIsVideoReady(true);
						};
					}
					setIsAllowed(true);
				} catch (err) {
					console.error("カメラアクセスエラー:", err);
					setIsAllowed(false);
				}
			};

			setupCamera();
			
			// クリーンアップ
			return () => {
				if (videoRef.current) {
					videoRef.current.onloadedmetadata = null;
					videoRef.current.onloadeddata = null;
					if (videoRef.current.srcObject) {
						const mediaStream = videoRef.current.srcObject as MediaStream;
						mediaStream.getTracks().forEach(track => track.stop());
					}
				}
				setIsVideoReady(false);
			};
		}
	}, [videoRef, videoFile]);

	return {
		isAllowed,
		isVideoReady,
	}
};

const useHandpose = (
	videoRef: RefObject<HTMLVideoElement>,
	canvasRef: RefObject<HTMLCanvasElement>,
	isVideoReady: boolean,
) => {
	const [isLoading, setIsLoading] = useState(false);
	const [model, setModel] = useState<handPoseDetection.HandDetector>();

	useEffect(() => {
		const loadHandpose = async () => {
			try {
				// TensorFlowバックエンドの準備を確認
				await tf.ready();
				
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
			if (!isVideoReady) return; // 動画が準備できていなければ検出しない
			
			try {
				// 動画のサイズチェック
				if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
					console.log("動画サイズが無効です。スキップします。");
					return;
				}
				
				// キャンバスサイズを動画サイズに合わせる
				if (canvasRef.current) {
					canvasRef.current.width = videoRef.current.videoWidth;
					canvasRef.current.height = videoRef.current.videoHeight;
				}

				const hands = await model.estimateHands(videoRef.current);

				if (!canvasRef.current) return;
				const ctx = canvasRef.current.getContext("2d");
				if (!ctx) return;

				ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

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
	}, [model, videoRef, canvasRef, isVideoReady]);

	return {
		isLoading,
	}
};

const useFaceDetection = (
	videoRef: RefObject<HTMLVideoElement>,
	canvasRef: RefObject<HTMLCanvasElement>,
	isVideoReady: boolean,
) => {
	const [isLoading, setIsLoading] = useState(false);
	const [model, setModel] = useState<faceLandmarksDetection.FaceLandmarksDetector>();

	useEffect(() => {
		const loadFaceDetection = async () => {
			try {
				// TensorFlowバックエンドの準備を確認
				await tf.ready();
				
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
			if (!isVideoReady) return; // 動画が準備できていなければ検出しない
			
			try {
				// 動画のサイズチェック
				if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
					console.log("動画サイズが無効です。スキップします。");
					return;
				}
				
				// キャンバスサイズを動画サイズに合わせる
				canvasRef.current.width = videoRef.current.videoWidth;
				canvasRef.current.height = videoRef.current.videoHeight;
				
				const faces = await model.estimateFaces(videoRef.current);
				
				const ctx = canvasRef.current.getContext("2d");
				if (!ctx) return;

				ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

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
	}, [model, videoRef, canvasRef, isVideoReady]);

	return {
		isLoading,
	}
};

const usePoseDetection = (
	videoRef: RefObject<HTMLVideoElement>,
	canvasRef: RefObject<HTMLCanvasElement>,
	isVideoReady: boolean,
) => {
	const [isLoading, setIsLoading] = useState(false);
	const [model, setModel] = useState<poseDetection.PoseDetector>();

	useEffect(() => {
		const loadPoseDetection = async () => {
			try {
				// TensorFlowバックエンドの準備を確認
				await tf.ready();
				
				const model = poseDetection.SupportedModels.MoveNet;
				// 複数人検出用のモデルを設定
				const detector = await poseDetection.createDetector(
					model,
					{
						modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
						enableSmoothing: true, // スムージングを有効化
					}
				);
				setModel(detector);
			} catch (error) {
				console.error("ポーズ検出モデルの読み込みエラー:", error);
			}
		};

		setIsLoading(true);
		loadPoseDetection().finally(() => {
			setIsLoading(false);
		});
	}, []);

	useEffect(() => {
		// 骨格の接続定義
		const connections = [
			// 顔
			['nose', 'left_eye'],
			['nose', 'right_eye'],
			['left_eye', 'left_ear'],
			['right_eye', 'right_ear'],
			// 上半身
			['left_shoulder', 'right_shoulder'],
			['left_shoulder', 'left_elbow'],
			['right_shoulder', 'right_elbow'],
			['left_elbow', 'left_wrist'],
			['right_elbow', 'right_wrist'],
			// 下半身
			['left_shoulder', 'left_hip'],
			['right_shoulder', 'right_hip'],
			['left_hip', 'right_hip'],
			['left_hip', 'left_knee'],
			['right_hip', 'right_knee'],
			['left_knee', 'left_ankle'],
			['right_knee', 'right_ankle'],
		];

		// 各ポーズに異なる色を割り当てるための配列
		const colors = [
			{ point: "lime", line: "aqua" },      // 1人目: 薄緑と水色
			{ point: "magenta", line: "yellow" },  // 2人目: マゼンタと黄色
			{ point: "orange", line: "red" },      // 3人目: オレンジと赤
			{ point: "cyan", line: "blue" },       // 4人目: シアンと青
			{ point: "white", line: "green" }      // 5人目: 白と緑
		];

		const detect = async () => {
			if (!model) return;
			if (!videoRef.current) return;
			if (!canvasRef.current) return;
			if (!isVideoReady) return; // 動画が準備できていなければ検出しない
			
			try {
				// 動画のサイズチェック
				if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
					console.log("動画サイズが無効です。スキップします。");
					return;
				}
				
				// キャンバスサイズを動画サイズに合わせる
				canvasRef.current.width = videoRef.current.videoWidth;
				canvasRef.current.height = videoRef.current.videoHeight;
				
				const poses = await model.estimatePoses(videoRef.current);
				
				const ctx = canvasRef.current.getContext("2d");
				if (!ctx) return;

				ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

				// 検出された人数を表示
				ctx.font = "16px Arial";
				ctx.fillStyle = "white";
				ctx.fillText(`検出された人数: ${poses.length}人`, 20, 30);

				// 各検出されたポーズを処理
				for (let i = 0; i < poses.length; i++) {
					const pose = poses[i];
					const keypoints = pose.keypoints;
					const colorSet = colors[i % colors.length]; // 人数に応じて色を循環使用

					// キーポイントの描画
					for (const keypoint of keypoints) {
						if (keypoint.score && keypoint.score > 0.3) { // 信頼度が30%以上のポイントのみ描画
							const { x, y } = keypoint;
							ctx.beginPath();
							ctx.arc(x, y, 6, 0, 3 * Math.PI);
							ctx.fillStyle = colorSet.point;
							ctx.fill();
						}
					}
					
					// 骨格の線を描画
					ctx.lineWidth = 3;
					ctx.strokeStyle = colorSet.line;
					
					for (const [from, to] of connections) {
						const fromPoint = keypoints.find(kp => kp.name === from);
						const toPoint = keypoints.find(kp => kp.name === to);
						
						if (fromPoint && toPoint && 
							fromPoint.score && toPoint.score && 
							fromPoint.score > 0.3 && toPoint.score > 0.3) {
							ctx.beginPath();
							ctx.moveTo(fromPoint.x, fromPoint.y);
							ctx.lineTo(toPoint.x, toPoint.y);
							ctx.stroke();
						}
					}
					
					// 信頼度スコアの表示
					ctx.font = "16px Arial";
					ctx.fillStyle = colorSet.point;
					const score = Math.round((pose.score || 0) * 100) / 100;
					// 各人のスコアを表示（少しずつ位置をずらす）
					ctx.fillText(`ID ${i+1} 信頼度: ${score}`, 20, 60 + i * 25);
				}
			} catch (error) {
				console.error("ポーズ検出エラー:", error);
			}
		};

		const interval = setInterval(detect, 100); // 0.1秒ごとに検出
		return () => clearInterval(interval);
	}, [model, videoRef, canvasRef, isVideoReady]);

	return {
		isLoading,
	}
};

function App() {
	const videoRef = useRef<HTMLVideoElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [videoFile, setVideoFile] = useState<File | null>(null);
	const { isAllowed, isVideoReady } = useVideo(videoRef, videoFile);
	const [activeTab, setActiveTab] = useState<TabType>("face");
	
	const { isLoading: isHandLoading } = useHandpose(
		activeTab === "hand" ? videoRef : { current: null },
		activeTab === "hand" ? canvasRef : { current: null },
		isVideoReady
	);
	
	const { isLoading: isFaceLoading } = useFaceDetection(
		activeTab === "face" ? videoRef : { current: null },
		activeTab === "face" ? canvasRef : { current: null },
		isVideoReady
	);

	const { isLoading: isPoseLoading } = usePoseDetection(
		activeTab === "pose" ? videoRef : { current: null },
		activeTab === "pose" ? canvasRef : { current: null },
		isVideoReady
	);

	const isLoading = 
		activeTab === "hand" ? isHandLoading : 
		activeTab === "face" ? isFaceLoading : 
		isPoseLoading;

	// ファイル選択ハンドラー
	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = event.target.files;
		if (files && files.length > 0) {
			const file = files[0];
			// ファイルがビデオかどうかをチェック
			if (file.type.startsWith('video/')) {
				setVideoFile(file);
			} else {
				alert('動画ファイルを選択してください');
			}
		}
	};

	// ファイル選択をリセット
	const resetVideo = () => {
		setVideoFile(null);
	};

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
					playsInline
				/>
				<canvas
					style={{
						position: "absolute",
						top: 0,
						left: 0,
						width: "100%",
						height: "100%",
						objectFit: "cover",
						pointerEvents: "none", // キャンバスを操作不可に
					}}
					ref={canvasRef}
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
				}}>TensorFlow.jsを使った手と顔とポーズの検出</p>
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
				<button 
					onClick={() => setActiveTab("pose")}
					style={{ 
						padding: "8px 16px", 
						backgroundColor: activeTab === "pose" ? "rgba(52, 152, 219, 0.9)" : "rgba(0, 0, 0, 0.6)",
						color: "white",
						border: "none",
						borderRadius: "30px",
						cursor: "pointer",
						backdropFilter: "blur(4px)",
						boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
						display: "flex",
						alignItems: "center",
						fontWeight: activeTab === "pose" ? "bold" : "normal",
						transition: "all 0.3s ease",
					}}
				>
					<span style={{ marginRight: "6px" }}>🏃</span> ポーズ検出
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
					<strong>ヒント:</strong> {
						activeTab === "hand" ? "両手を画面内に表示すると、関節と骨格が検出されます。" : 
						activeTab === "face" ? "顔を画面内に表示すると、顔のランドマークが検出されます。" :
						"複数人の姿勢も検出できます。それぞれ異なる色で表示されます。"
					}
				</p>
			</div>

			{/* ファイル入力UI */}
			<div style={{ 
				position: "absolute",
				bottom: "80px",
				left: "50%",
				transform: "translateX(-50%)",
				zIndex: 10,
				padding: "10px 20px", 
				backgroundColor: "rgba(0, 0, 0, 0.6)",
				color: "white",
				borderRadius: "30px",
				backdropFilter: "blur(4px)",
				boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
				display: "flex",
				alignItems: "center",
				gap: "10px",
			}}>
				{videoFile ? (
					<>
						<span>{videoFile.name}</span>
						<button
							onClick={resetVideo}
							style={{
								backgroundColor: "rgba(220, 53, 69, 0.7)",
								color: "white",
								border: "none",
								borderRadius: "20px",
								padding: "6px 12px",
								cursor: "pointer",
								display: "flex",
								alignItems: "center",
								gap: "5px",
							}}
						>
							<span>✖</span> キャンセル
						</button>
					</>
				) : (
					<>
						<label
							htmlFor="video-upload"
							style={{
								backgroundColor: "rgba(52, 152, 219, 0.7)",
								color: "white",
								borderRadius: "20px",
								padding: "6px 12px",
								cursor: "pointer",
								display: "flex",
								alignItems: "center",
								gap: "5px",
							}}
						>
							<span>📁</span> 動画ファイルを選択
						</label>
						<input
							id="video-upload"
							type="file"
							accept="video/*"
							onChange={handleFileChange}
							style={{ display: "none" }}
						/>
					</>
				)}
			</div>

			{/* 動画読み込み状態の表示 */}
			{videoFile && !isVideoReady && (
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
					<p style={{ margin: "0", fontWeight: "bold" }}>動画を読み込み中...</p>
				</div>
			)}

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
