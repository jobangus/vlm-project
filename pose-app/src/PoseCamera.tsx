import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

// MediaPipe pose landmark connections for drawing the skeleton
const POSE_CONNECTIONS = PoseLandmarker.POSE_CONNECTIONS;

const PoseCamera: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const animationFrameRef = useRef<number>(0);
  const lastTimestampRef = useRef<number>(-1);
  const streamRef = useRef<MediaStream | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [poseDetected, setPoseDetected] = useState(false);

  // Initialize MediaPipe Pose Landmarker
  const initPoseLandmarker = useCallback(async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      poseLandmarkerRef.current = poseLandmarker;
      console.log("✅ MediaPipe Pose Landmarker initialized");
    } catch (err) {
      console.error("Failed to initialize PoseLandmarker:", err);
      setError("Failed to load pose detection model. Please refresh.");
    }
  }, []);

  // Start webcam
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        console.log("✅ Camera started");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      console.error("Camera access denied:", err);
      setError(
        "Camera access denied. Please allow camera access and refresh."
      );
    }
  }, []);

  // Draw pose landmarks on canvas
  const drawLandmarks = useCallback(
    (landmarks: any[], canvasCtx: CanvasRenderingContext2D) => {
      const drawingUtils = new DrawingUtils(canvasCtx);

      for (const poseLandmarks of landmarks) {
        // Draw skeleton connections
        drawingUtils.drawConnectors(poseLandmarks, POSE_CONNECTIONS, {
          color: "#00FF88",
          lineWidth: 3,
        });

        // Draw landmark points
        drawingUtils.drawLandmarks(poseLandmarks, {
          color: "#FF3366",
          lineWidth: 1,
          radius: 4,
        });
      }
    },
    []
  );

  // Detection loop
  const detectPose = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const poseLandmarker = poseLandmarkerRef.current;

    if (!video || !canvas || !poseLandmarker || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectPose);
      return;
    }

    // Resize canvas to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;

    // Clear previous frame
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Only call detectForVideo if timestamp has changed
    const timestamp = performance.now();
    if (timestamp !== lastTimestampRef.current) {
      lastTimestampRef.current = timestamp;

      const startTime = performance.now();
      const result = poseLandmarker.detectForVideo(video, timestamp);
      const endTime = performance.now();

      // Calculate FPS
      const inferenceTime = endTime - startTime;
      setFps(Math.round(1000 / Math.max(inferenceTime, 1)));

      // Draw results
      if (result.landmarks && result.landmarks.length > 0) {
        setPoseDetected(true);
        drawLandmarks(result.landmarks, canvasCtx);
      } else {
        setPoseDetected(false);
      }
    }

    animationFrameRef.current = requestAnimationFrame(detectPose);
  }, [drawLandmarks]);

  // Initialize everything on mount
  useEffect(() => {
    let isActive = true;
    const videoEl = videoRef.current;

    const init = async () => {
      setIsLoading(true);
      await initPoseLandmarker();
      if (!isActive) return;
      await startCamera();
      if (!isActive) return;
      setIsLoading(false);
    };

    init();

    return () => {
      isActive = false;

      // Cleanup
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      if (poseLandmarkerRef.current) {
        poseLandmarkerRef.current.close();
        poseLandmarkerRef.current = null;
      }

      if (streamRef.current) {
        const tracks = streamRef.current.getTracks();
        tracks.forEach((track) => track.stop());
        streamRef.current = null;
      }

      if (videoEl) {
        videoEl.srcObject = null;
      }
    };
  }, [initPoseLandmarker, startCamera]);

  // Start detection loop once loading is done
  useEffect(() => {
    if (!isLoading && !error) {
      animationFrameRef.current = requestAnimationFrame(detectPose);
    }
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isLoading, error, detectPose]);

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>📸 Pose Coach</h1>
        <div style={styles.stats}>
          <span style={styles.fpsBadge}>⚡ {fps} FPS</span>
          <span
            style={{
              ...styles.statusBadge,
              backgroundColor: poseDetected ? "#00FF88" : "#FF3366",
            }}
          >
            {poseDetected ? "✅ Pose Detected" : "❌ No Pose"}
          </span>
        </div>
      </div>

      {/* Camera + Canvas overlay */}
      <div style={styles.cameraContainer}>
        {isLoading && (
          <div style={styles.loadingOverlay}>
            <p style={styles.loadingText}>Loading pose detection model...</p>
          </div>
        )}

        {error && (
          <div style={styles.errorOverlay}>
            <p style={styles.errorText}>⚠️ {error}</p>
          </div>
        )}

        <video
          ref={videoRef}
          style={styles.video}
          playsInline
          muted
        />
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    backgroundColor: "#1a1a2e",
    minHeight: "100vh",
    color: "white",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    width: "100%",
    maxWidth: "800px",
    padding: "16px 20px",
    boxSizing: "border-box",
  },
  title: {
    margin: 0,
    fontSize: "1.5rem",
  },
  stats: {
    display: "flex",
    gap: "8px",
    alignItems: "center",
  },
  fpsBadge: {
    backgroundColor: "#16213e",
    padding: "4px 10px",
    borderRadius: "12px",
    fontSize: "0.85rem",
  },
  statusBadge: {
    padding: "4px 10px",
    borderRadius: "12px",
    fontSize: "0.85rem",
    color: "#1a1a2e",
    fontWeight: 600,
  },
  cameraContainer: {
    position: "relative",
    width: "100%",
    maxWidth: "800px",
    borderRadius: "12px",
    overflow: "hidden",
    boxShadow: "0 4px 30px rgba(0, 0, 0, 0.5)",
  },
  video: {
    width: "100%",
    display: "block",
    transform: "scaleX(-1)", // Mirror for selfie view
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    transform: "scaleX(-1)", // Mirror to match video
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.7)",
    zIndex: 10,
  },
  loadingText: {
    fontSize: "1.2rem",
    color: "#00FF88",
  },
  errorOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.7)",
    zIndex: 10,
  },
  errorText: {
    fontSize: "1.2rem",
    color: "#FF3366",
    textAlign: "center",
    padding: "0 20px",
  },
};

export default PoseCamera;
