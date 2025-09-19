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
	const [emotion, setEmotion] = useState<string>("検出中...");
	const [irisPosition, setIrisPosition] = useState<{ leftEye: string; rightEye: string; overall: string }>({
		leftEye: "検出中",
		rightEye: "検出中",
		overall: "虹彩を検出中..."
	});

	useEffect(() => {
		const loadFaceDetection = async () => {
			try {
				// TensorFlowバックエンドの準備を確認
				await tf.ready();
				
				const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
				const detectorConfig = {
					runtime: 'tfjs',
					refineLandmarks: true, // 詳細なランドマークを取得
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

	// 虹彩位置検出関数
	const detectIrisPosition = (keypoints: faceLandmarksDetection.Keypoint[]) => {
		// MediaPipe FaceMeshの虹彩関連のランドマーク
		// 左目虹彩: 468-472
		// 右目虹彩: 473-477
		const leftIrisCenter = keypoints[468]; // 左虹彩の中心
		const rightIrisCenter = keypoints[473]; // 右虹彩の中心

		// 目の端のランドマーク
		const leftEyeInner = keypoints[133]; // 左目内側
		const leftEyeOuter = keypoints[33];  // 左目外側
		const rightEyeInner = keypoints[362]; // 右目内側
		const rightEyeOuter = keypoints[263]; // 右目外側

		// 目の上下のランドマーク
		const leftEyeTop = keypoints[159];
		const leftEyeBottom = keypoints[145];
		const rightEyeTop = keypoints[386];
		const rightEyeBottom = keypoints[374];

		if (!leftIrisCenter || !rightIrisCenter) {
			return {
				leftEye: "検出中",
				rightEye: "検出中",
				overall: "虹彩を検出中..."
			};
		}

		// 左目の虹彩位置を計算
		const leftEyeWidth = Math.abs(leftEyeOuter.x - leftEyeInner.x);
		const leftEyeHeight = Math.abs(leftEyeBottom.y - leftEyeTop.y);
		const leftIrisRelativeX = (leftIrisCenter.x - leftEyeInner.x) / leftEyeWidth;
		const leftIrisRelativeY = (leftIrisCenter.y - leftEyeTop.y) / leftEyeHeight;

		// 右目の虹彩位置を計算
		const rightEyeWidth = Math.abs(rightEyeOuter.x - rightEyeInner.x);
		const rightEyeHeight = Math.abs(rightEyeBottom.y - rightEyeTop.y);
		const rightIrisRelativeX = (rightIrisCenter.x - rightEyeInner.x) / rightEyeWidth;
		const rightIrisRelativeY = (rightIrisCenter.y - rightEyeTop.y) / rightEyeHeight;

		// 左目の方向を判定
		let leftEyeDirection = "中央";
		if (leftIrisRelativeX < 0.35) {
			leftEyeDirection = "外側";
		} else if (leftIrisRelativeX > 0.65) {
			leftEyeDirection = "内側";
		}

		if (leftIrisRelativeY < 0.35) {
			leftEyeDirection = leftEyeDirection === "中央" ? "上" : leftEyeDirection + "・上";
		} else if (leftIrisRelativeY > 0.65) {
			leftEyeDirection = leftEyeDirection === "中央" ? "下" : leftEyeDirection + "・下";
		}

		// 右目の方向を判定
		let rightEyeDirection = "中央";
		if (rightIrisRelativeX < 0.35) {
			rightEyeDirection = "内側";
		} else if (rightIrisRelativeX > 0.65) {
			rightEyeDirection = "外側";
		}

		if (rightIrisRelativeY < 0.35) {
			rightEyeDirection = rightEyeDirection === "中央" ? "上" : rightEyeDirection + "・上";
		} else if (rightIrisRelativeY > 0.65) {
			rightEyeDirection = rightEyeDirection === "中央" ? "下" : rightEyeDirection + "・下";
		}

		// 全体的な視線方向を判定
		let overallDirection = "正面";
		const avgX = (leftIrisRelativeX + rightIrisRelativeX) / 2;
		const avgY = (leftIrisRelativeY + rightIrisRelativeY) / 2;

		if (avgX < 0.35) {
			overallDirection = "右";
		} else if (avgX > 0.65) {
			overallDirection = "左";
		}

		if (avgY < 0.35) {
			overallDirection = overallDirection === "正面" ? "上" : overallDirection + "上";
		} else if (avgY > 0.65) {
			overallDirection = overallDirection === "正面" ? "下" : overallDirection + "下";
		}

		return {
			leftEye: leftEyeDirection,
			rightEye: rightEyeDirection,
			overall: `視線: ${overallDirection}`
		};
	};

	// 表情検出関数
	const detectEmotion = (keypoints: faceLandmarksDetection.Keypoint[]) => {
		// 主要なランドマークのインデックス（MediaPipe Facemeshの場合）
		// 口角のインデックス
		const leftMouthCorner = keypoints.find(kp => kp.name === 'leftMouthCorner') || keypoints[61];
		const rightMouthCorner = keypoints.find(kp => kp.name === 'rightMouthCorner') || keypoints[291];
		
		// 口の上下のインデックス
		const upperLipTop = keypoints.find(kp => kp.name === 'upperLipTop') || keypoints[13];
		const lowerLipBottom = keypoints.find(kp => kp.name === 'lowerLipBottom') || keypoints[14];
		
		// 目のインデックス
		const leftEyeTop = keypoints.find(kp => kp.name === 'leftEyeTop') || keypoints[159];
		const leftEyeBottom = keypoints.find(kp => kp.name === 'leftEyeBottom') || keypoints[145];
		const rightEyeTop = keypoints.find(kp => kp.name === 'rightEyeTop') || keypoints[386];
		const rightEyeBottom = keypoints.find(kp => kp.name === 'rightEyeBottom') || keypoints[374];
		
		// 眉毛のインデックス
		const leftEyebrowTop = keypoints.find(kp => kp.name === 'leftEyebrowTop') || keypoints[66];
		const rightEyebrowTop = keypoints.find(kp => kp.name === 'rightEyebrowTop') || keypoints[296];
		const leftEyebrowBottom = keypoints.find(kp => kp.name === 'leftEyebrowBottom') || keypoints[65];
		const rightEyebrowBottom = keypoints.find(kp => kp.name === 'rightEyebrowBottom') || keypoints[295];
		
		// 笑顔の検出（口角が上がっているか）
		const mouthCenter = {
			x: (leftMouthCorner.x + rightMouthCorner.x) / 2,
			y: (leftMouthCorner.y + rightMouthCorner.y) / 2
		};
		
		// 口の開き具合
		const mouthOpenness = Math.abs(upperLipTop.y - lowerLipBottom.y);
		
		// 口角の上がり具合（yが小さいほど上）
		const leftCornerHeight = mouthCenter.y - leftMouthCorner.y;
		const rightCornerHeight = mouthCenter.y - rightMouthCorner.y;
		const averageCornerHeight = (leftCornerHeight + rightCornerHeight) / 2;
		
		// 口角の下がり具合（yが大きいほど下）
		const leftCornerDropping = leftMouthCorner.y - mouthCenter.y;
		const rightCornerDropping = rightMouthCorner.y - mouthCenter.y;
		const averageCornerDropping = (leftCornerDropping + rightCornerDropping) / 2;
		
		// 目の開き具合
		const leftEyeOpenness = Math.abs(leftEyeTop.y - leftEyeBottom.y);
		const rightEyeOpenness = Math.abs(rightEyeTop.y - rightEyeBottom.y);
		const averageEyeOpenness = (leftEyeOpenness + rightEyeOpenness) / 2;
		
		// 眉の位置と動き
		const eyebrowHeight = (leftEyebrowTop.y + rightEyebrowTop.y) / 2;
		const eyesHeight = (leftEyeTop.y + rightEyeTop.y) / 2;
		const eyebrowEyeDistance = eyesHeight - eyebrowHeight;
		
		// 眉間の寄り具合
		const eyebrowInnerDistance = Math.abs(leftEyebrowBottom.x - rightEyebrowBottom.x);
		
		// ウインクの検出
		const isLeftWink = leftEyeOpenness < rightEyeOpenness * 0.4;
		const isRightWink = rightEyeOpenness < leftEyeOpenness * 0.4;
		
		// 大喜びの検出（眉が上がり、口が大きく開き、口角も上がっている）
		if (eyebrowEyeDistance > 15 && mouthOpenness > 15 && averageCornerHeight > 8) {
			return "大喜び 🤩";
		}
		
		// 笑顔の判定（口角が上がっていて、目が少し細くなっている）
		if (averageCornerHeight > 5 && mouthOpenness > 3) {
			return "笑顔 😊";
		}
		
		// ウインクの検出
		if (isLeftWink && !isRightWink) {
			return "左ウインク 😉";
		}
		if (isRightWink && !isLeftWink) {
			return "右ウインク 😉";
		}
		
		// 驚きの判定（目が大きく開いていて、口も開いている、眉も上がっている）
		if (averageEyeOpenness > 15 && mouthOpenness > 10 && eyebrowEyeDistance > 12) {
			return "驚き 😲";
		}
		
		// 悲しみの判定（口角が下がっている、目が細い）
		if (averageCornerDropping > 5 && averageEyeOpenness < 10) {
			return "悲しみ 😢";
		}
		
		// 怒りの判定（眉間が寄っている、口が開いている）
		if (eyebrowInnerDistance < 60 && mouthOpenness > 5) {
			return "怒り 😠";
		}
		
		// 眠そうな表情（目が半分閉じている）
		if (averageEyeOpenness < 7) {
			return "眠い 😴";
		}
		
		// 困惑/混乱（眉が上がっている、口が少し開いている）
		if (eyebrowEyeDistance > 10 && mouthOpenness > 3 && mouthOpenness < 10) {
			return "困惑 🤔";
		}
		
		// 真面目な表情（口がほぼ閉じている、表情があまり変化していない）
		if (mouthOpenness < 5 && Math.abs(averageCornerHeight) < 3) {
			return "真面目 😐";
		}
		
		// 十分な特徴がない場合
		return "中立 😶";
	};

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
					
					// 表情と虹彩位置を検出
					if (keypoints.length > 0) {
						const detectedEmotion = detectEmotion(keypoints);
						setEmotion(detectedEmotion);

						// 虹彩位置を検出
						const detectedIris = detectIrisPosition(keypoints);
						setIrisPosition(detectedIris);
					}
					
					// ランドマークの描画
					for (let j = 0; j < keypoints.length; j++) {
						const { x, y } = keypoints[j];
						ctx.beginPath();

						// 虹彩関連のランドマークを強調表示
						if (j >= 468 && j <= 477) {
							// 虹彩のランドマーク（468-477）を黄色で大きく表示
							ctx.arc(x, y, 3, 0, 3 * Math.PI);
							ctx.fillStyle = "yellow";
						} else if ([33, 133, 159, 145, 263, 362, 386, 374].includes(j)) {
							// 目の端と上下のランドマークを青で表示
							ctx.arc(x, y, 2, 0, 3 * Math.PI);
							ctx.fillStyle = "cyan";
						} else {
							// その他のランドマーク
							ctx.arc(x, y, 1, 0, 3 * Math.PI);
							ctx.fillStyle = "red";
						}
						ctx.fill();
					}
					
					// 顔の中心と上部を計算
					let faceTop = 50; // デフォルト値
					let faceCenter = canvasRef.current.width / 2; // デフォルト値
					
					// 髪の生え際あたりのポイント (10番目あたり)を探す
					const foreheadPoint = keypoints.find(kp => kp.name === 'foreheadMid') || keypoints[10];
					if (foreheadPoint) {
						faceTop = foreheadPoint.y - 40; // 額の少し上
						faceCenter = foreheadPoint.x;
					} else {
						// 目の位置からも推定可能
						const leftEye = keypoints.find(kp => kp.name === 'leftEye') || keypoints[159];
						const rightEye = keypoints.find(kp => kp.name === 'rightEye') || keypoints[386];
						
						if (leftEye && rightEye) {
							faceCenter = (leftEye.x + rightEye.x) / 2;
							faceTop = Math.min(leftEye.y, rightEye.y) - 50;
						}
					}
					
					// 表示位置が画面外にならないよう調整
					faceTop = Math.max(30, faceTop);
					
					// 表情の表示 - 顔の上部に表示
					const emotionText = `${emotion}`;
					ctx.font = "bold 24px Arial";

					// テキストの幅を取得してセンタリング
					const textWidth = ctx.measureText(emotionText).width;
					const textX = faceCenter - (textWidth / 2);

					// テキストに縁取りを追加して視認性を向上
					ctx.strokeStyle = "black";
					ctx.lineWidth = 3;
					ctx.strokeText(emotionText, textX, faceTop);

					// テキスト
					ctx.fillStyle = "white";
					ctx.fillText(emotionText, textX, faceTop);

					// 虹彩位置の表示 - 表情の下に表示
					const irisText = irisPosition.overall;
					ctx.font = "bold 20px Arial";

					const irisTextWidth = ctx.measureText(irisText).width;
					const irisTextX = faceCenter - (irisTextWidth / 2);
					const irisTextY = faceTop + 30;

					// 縁取り
					ctx.strokeStyle = "black";
					ctx.lineWidth = 3;
					ctx.strokeText(irisText, irisTextX, irisTextY);

					// テキスト
					ctx.fillStyle = "yellow";
					ctx.fillText(irisText, irisTextX, irisTextY);

					// 各目の詳細（小さく表示）
					ctx.font = "14px Arial";
					const detailText = `左目: ${irisPosition.leftEye} | 右目: ${irisPosition.rightEye}`;
					const detailTextWidth = ctx.measureText(detailText).width;
					const detailTextX = faceCenter - (detailTextWidth / 2);
					const detailTextY = irisTextY + 20;

					ctx.strokeStyle = "black";
					ctx.lineWidth = 2;
					ctx.strokeText(detailText, detailTextX, detailTextY);

					ctx.fillStyle = "white";
					ctx.fillText(detailText, detailTextX, detailTextY);
				}
				
				// 顔が検出されない場合
				if (faces.length === 0) {
					setEmotion("顔が見つかりません");
					
					// 顔が見つからないメッセージを画面中央に表示
					const noFaceText = `${emotion}`;
					ctx.font = "bold 24px Arial";
					
					// テキストの幅を取得してセンタリング
					const textWidth = ctx.measureText(noFaceText).width;
					const textX = canvasRef.current.width / 2 - (textWidth / 2);
					const textY = canvasRef.current.height / 2;
					
					// テキストに縁取りを追加して視認性を向上
					ctx.strokeStyle = "black";
					ctx.lineWidth = 3;
					ctx.strokeText(noFaceText, textX, textY);
					
					// テキスト
					ctx.fillStyle = "white";
					ctx.fillText(noFaceText, textX, textY);
				}
			} catch (error) {
				console.error("顔の検出エラー:", error);
			}
		};

		const interval = setInterval(detect, 100); // 0.1秒ごとに検出
		return () => clearInterval(interval);
	}, [model, videoRef, canvasRef, isVideoReady, emotion, irisPosition]);

	return {
		isLoading,
		emotion,
		irisPosition,
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
	
	const { isLoading: isFaceLoading, emotion: faceEmotion, irisPosition } = useFaceDetection(
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

	// ビューポートの幅を監視するためのstate
	const [windowWidth, setWindowWidth] = useState(window.innerWidth);

	// ウィンドウサイズの変更を監視
	useEffect(() => {
		const handleResize = () => {
			setWindowWidth(window.innerWidth);
		};
		
		window.addEventListener('resize', handleResize);
		return () => {
			window.removeEventListener('resize', handleResize);
		};
	}, []);

	// モバイル表示かどうかの判定
	const isMobile = windowWidth <= 768;

	// 表情一覧
	const emotionList = [
		{ name: "大喜び", emoji: "🤩" },
		{ name: "笑顔", emoji: "😊" },
		{ name: "左/右ウインク", emoji: "😉" },
		{ name: "驚き", emoji: "😲" },
		{ name: "悲しみ", emoji: "😢" },
		{ name: "怒り", emoji: "😠" },
		{ name: "眠い", emoji: "😴" },
		{ name: "困惑", emoji: "🤔" },
		{ name: "真面目", emoji: "😐" },
		{ name: "中立", emoji: "😶" },
	];

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
				top: isMobile ? "10px" : "20px",
				left: isMobile ? "10px" : "20px",
				zIndex: 10,
				color: "white",
				textShadow: "0 2px 4px rgba(0,0,0,0.5)",
			}}>
				<h1 style={{
					margin: "0 0 4px 0",
					fontSize: isMobile ? "18px" : "24px",
					fontWeight: "bold",
				}}>AI姿勢検出デモ</h1>
				<p style={{
					margin: 0,
					fontSize: isMobile ? "12px" : "14px",
					opacity: 0.8,
				}}>TensorFlow.jsを使った手と顔とポーズの検出</p>
			</div>

			{/* フローティングタブ切り替え */}
			<div style={{ 
				position: "absolute",
				top: isMobile ? "10px" : "20px",
				right: isMobile ? "10px" : "20px",
				zIndex: 10,
				display: "flex",
				flexDirection: isMobile ? "column" : "row",
				gap: isMobile ? "8px" : "12px",
			}}>
				<button 
					onClick={() => setActiveTab("hand")}
					style={{ 
						padding: isMobile ? "6px 12px" : "8px 16px", 
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
						fontSize: isMobile ? "13px" : "15px",
						transition: "all 0.3s ease",
					}}
				>
					<span style={{ marginRight: "6px" }}>✋</span> 手の検出
				</button>
				<button 
					onClick={() => setActiveTab("face")}
					style={{ 
						padding: isMobile ? "6px 12px" : "8px 16px", 
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
						fontSize: isMobile ? "13px" : "15px",
						transition: "all 0.3s ease",
					}}
				>
					<span style={{ marginRight: "6px" }}>😊</span> 顔の検出
				</button>
				<button 
					onClick={() => setActiveTab("pose")}
					style={{ 
						padding: isMobile ? "6px 12px" : "8px 16px", 
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
						fontSize: isMobile ? "13px" : "15px",
						transition: "all 0.3s ease",
					}}
				>
					<span style={{ marginRight: "6px" }}>🏃</span> ポーズ検出
				</button>
			</div>

			{/* 表情一覧フローティングUI - 顔検出タブでのみ表示 */}
			{activeTab === "face" && (
				<div style={{ 
					position: "absolute",
					top: "50%",
					right: isMobile ? "5px" : "10px",
					transform: "translateY(-50%)",
					zIndex: 10,
					padding: isMobile ? "6px 8px" : "8px 10px", 
					backgroundColor: "rgba(0, 0, 0, 0.6)",
					color: "white",
					borderRadius: "8px",
					backdropFilter: "blur(4px)",
					boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
					display: "flex",
					flexDirection: "column",
					gap: isMobile ? "4px" : "6px",
					maxHeight: "50vh",
					maxWidth: isMobile ? "100px" : "120px",
					overflowY: "auto",
				}}>
					<div style={{ 
						fontWeight: "bold", 
						fontSize: isMobile ? "11px" : "13px",
						textAlign: "center",
						marginBottom: "2px",
					}}>
						表情一覧
					</div>
					{emotionList.map((emotion, index) => (
						<div key={index} style={{ 
							display: "flex", 
							alignItems: "center", 
							gap: "4px",
							backgroundColor: faceEmotion.includes(emotion.name) ? "rgba(52, 152, 219, 0.3)" : "transparent",
							padding: "2px 4px",
							borderRadius: "4px",
						}}>
							<span style={{ fontSize: isMobile ? "12px" : "14px" }}>{emotion.emoji}</span>
							<span style={{ fontSize: isMobile ? "10px" : "11px" }}>{emotion.name}</span>
						</div>
					))}
				</div>
			)}
			
			{/* フローティングヒント */}
			<div style={{ 
				position: "absolute",
				bottom: isMobile ? "140px" : "20px",
				left: "50%",
				transform: "translateX(-50%)",
				zIndex: 10,
				padding: isMobile ? "8px 16px" : "10px 20px", 
				backgroundColor: "rgba(0, 0, 0, 0.6)",
				color: "white",
				borderRadius: "30px",
				backdropFilter: "blur(4px)",
				boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
				maxWidth: isMobile ? "90%" : "80%",
				textAlign: "center",
			}}>
				<p style={{ margin: "0", fontSize: isMobile ? "13px" : "15px" }}>
					<strong>ヒント:</strong> {
						activeTab === "hand" ? "両手を画面内に表示すると、関節と骨格が検出されます。" :
						activeTab === "face" ? `顔を画面内に表示すると、虹彩の位置と表情が検出されます。(${faceEmotion} / ${irisPosition.overall})` :
						"複数人の姿勢も検出できます。それぞれ異なる色で表示されます。"
					}
				</p>
			</div>

			{/* ファイル入力UI */}
			<div style={{ 
				position: "absolute",
				bottom: isMobile ? "80px" : "80px",
				left: "50%",
				transform: "translateX(-50%)",
				zIndex: 10,
				padding: isMobile ? "8px 16px" : "10px 20px", 
				backgroundColor: "rgba(0, 0, 0, 0.6)",
				color: "white",
				borderRadius: "30px",
				backdropFilter: "blur(4px)",
				boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
				display: "flex",
				flexDirection: isMobile && videoFile ? "column" : "row",
				alignItems: "center",
				gap: "10px",
				width: isMobile ? "auto" : "auto",
				maxWidth: isMobile ? "90%" : "auto",
			}}>
				{videoFile ? (
					<>
						<span style={{ 
							fontSize: isMobile ? "13px" : "15px",
							whiteSpace: "nowrap",
							overflow: "hidden",
							textOverflow: "ellipsis",
							maxWidth: isMobile ? "200px" : "300px",
						}}>{videoFile.name}</span>
						<button
							onClick={resetVideo}
							style={{
								backgroundColor: "rgba(220, 53, 69, 0.7)",
								color: "white",
								border: "none",
								borderRadius: "20px",
								padding: isMobile ? "5px 10px" : "6px 12px",
								cursor: "pointer",
								display: "flex",
								alignItems: "center",
								gap: "5px",
								fontSize: isMobile ? "13px" : "15px",
								marginTop: isMobile ? "8px" : "0",
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
								padding: isMobile ? "5px 10px" : "6px 12px",
								cursor: "pointer",
								display: "flex",
								alignItems: "center",
								gap: "5px",
								fontSize: isMobile ? "13px" : "15px",
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
					padding: isMobile ? "15px 20px" : "20px 30px",
					borderRadius: "8px",
					backdropFilter: "blur(10px)",
					boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
					textAlign: "center",
					width: isMobile ? "80%" : "auto",
					minWidth: isMobile ? "auto" : "300px",
				}}>
					<div style={{ fontSize: isMobile ? "26px" : "32px", marginBottom: "10px" }}>📷</div>
					<p style={{ margin: "0", fontWeight: "bold", fontSize: isMobile ? "15px" : "16px" }}>カメラへのアクセスが必要です</p>
					<p style={{ margin: "8px 0 0 0", fontSize: isMobile ? "13px" : "14px" }}>このアプリはカメラを使用して手と顔を検出します</p>
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
					padding: isMobile ? "15px 20px" : "20px 30px",
					borderRadius: "8px",
					backdropFilter: "blur(10px)",
					boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
					textAlign: "center",
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					justifyContent: "center",
					width: isMobile ? "auto" : "auto",
					minWidth: isMobile ? "auto" : "200px",
				}}>
					<div style={{
						border: "3px solid rgba(255, 255, 255, 0.1)",
						borderTop: "3px solid #fff",
						borderRadius: "50%",
						width: isMobile ? "25px" : "30px",
						height: isMobile ? "25px" : "30px",
						animation: "spin 1s linear infinite",
						marginBottom: "12px",
					}} />
					<p style={{ margin: "0", fontWeight: "bold", fontSize: isMobile ? "14px" : "16px" }}>モデルを読み込み中...</p>
				</div>
			)}
			
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
					padding: isMobile ? "15px 20px" : "20px 30px",
					borderRadius: "8px",
					backdropFilter: "blur(10px)",
					boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
					textAlign: "center",
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					justifyContent: "center",
					width: isMobile ? "auto" : "auto",
					minWidth: isMobile ? "auto" : "200px",
				}}>
					<div style={{
						border: "3px solid rgba(255, 255, 255, 0.1)",
						borderTop: "3px solid #fff",
						borderRadius: "50%",
						width: isMobile ? "25px" : "30px",
						height: isMobile ? "25px" : "30px",
						animation: "spin 1s linear infinite",
						marginBottom: "12px",
					}} />
					<p style={{ margin: "0", fontWeight: "bold", fontSize: isMobile ? "14px" : "16px" }}>動画を読み込み中...</p>
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
				@media (max-width: 768px) {
					button, label {
						-webkit-tap-highlight-color: transparent;
					}
					button:active, label:active {
						opacity: 0.8;
					}
				}
			`}</style>
		</div>
	);
}

export default App;
