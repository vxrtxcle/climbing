import React, { useRef, useState, useEffect } from "react";
import * as posedetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    const initializeBackend = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
    };
    initializeBackend();
  }, []);

  const handleVideoInput = async (event) => {
    const file = event.target.files[0];

    if (file) {
      const video = document.createElement("video");
      video.src = URL.createObjectURL(file);
      video.loop = false;
      video.muted = true;
      video.play();

      video.onloadeddata = async () => {
        const stream = video.captureStream();
        videoRef.current.srcObject = stream;

        // Initialize MoveNet model
        const detectorConfig = {
          modelType: posedetection.movenet.modelType.SINGLEPOSE_THUNDER,
        }
        const detector = await posedetection.createDetector(
          posedetection.SupportedModels.MoveNet, detectorConfig
        );

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const processedChunks = [];
        const mediaRecorder = new MediaRecorder(canvas.captureStream());

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            processedChunks.push(event.data);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(processedChunks, { type: "video/mp4" });
          const url = URL.createObjectURL(blob);

          const a = document.createElement("a");
          a.style.display = "none";
          a.href = url;
          a.download = "movenet_output.mp4";
          document.body.appendChild(a);
          a.click();
          URL.revokeObjectURL(url);
        };

        mediaRecorder.start();

        setProcessing(true);

        const drawSkeleton = (keypoints, ctx) => {
          const adjacentPairs = posedetection.util.getAdjacentPairs(
            posedetection.SupportedModels.MoveNet
          );

          adjacentPairs.forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];

            if (kp1.score > 0.5 && kp2.score > 0.5) {
              var average = (kp1.score + kp2.score) / 2;
              ctx.beginPath();
              ctx.moveTo(kp1.x, kp1.y);
              ctx.lineTo(kp2.x, kp2.y);
              ctx.strokeStyle = "rgb(" + (765 - (average)*1020) + "," + (average*1020-510) + ", 0)";
              ctx.lineWidth = 2;
              ctx.stroke();
            }
            if (kp1.score > 0.75 && kp2.score > 0.75) {
              ctx.beginPath();
              ctx.moveTo(kp1.x, kp1.y);
              ctx.lineTo(kp2.x, kp2.y);
              ctx.strokeStyle = "green";
              ctx.lineWidth = 2;
              ctx.stroke();
            }
            
    
          });
        };

        const processFrame = async () => {
          if (!video.paused && !video.ended) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const poses = await detector.estimatePoses(video);

            // Draw detected keypoints and skeleton
            poses.forEach((pose) => {
              drawSkeleton(pose.keypoints, ctx);

              pose.keypoints.forEach(({ x, y, score }) => {
                console.log("score: ", score);
                if (score > 0.5) {
                  ctx.beginPath();
                  ctx.arc(x, y, 3.2, 0, 2 * Math.PI);
                  ctx.fillStyle = "rgb(" + (765 - (score)*1020) + "," + (score*1020-510) + ", 0)";
                  ctx.fill();
                }
                if (score > 0.75) {
                  ctx.beginPath();
                  ctx.arc(x, y, 3.2, 0, 2 * Math.PI);
                  ctx.fillStyle = "green";
                  ctx.fill();
                }
              });
            });

            requestAnimationFrame(processFrame);
          } else {
            mediaRecorder.stop();
            setProcessing(false);
          }
        };

        processFrame();
      };
    }
  };

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-2xl font-bold">MoveNet Video Processor</h1>

      <input
        type="file"
        accept="video/*"
        onChange={handleVideoInput}
        className="p-2 border rounded"
        disabled={processing}
      />

      <div>
        <video
          ref={videoRef}
          controls
          className="w-full h-auto border rounded"
        ></video>
        <canvas ref={canvasRef} className="hidden"></canvas>
      </div>

      {processing && <p className="text-blue-500">Processing video...</p>}
    </div>
  );
};

export default App;
