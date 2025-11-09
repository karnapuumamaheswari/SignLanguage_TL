// Ensure the script runs after DOM is ready and avoid duplicate/global leaks
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const labelSpan = document.getElementById('label');
  const confSpan = document.getElementById('conf');
  const intervalInput = document.getElementById('interval');

  let stream = null;
  let timer = null;
  let pendingRequest = false;

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = stream;
      await video.play();
    } catch (e) {
      alert('Could not access webcam: ' + e.message);
    }
  }

  async function sendFrame() {
    if (!video || video.readyState < 2) return;
    // don't post another frame while a request is pending
    if (pendingRequest) return;
    // draw centered square from video
    const side = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, 224, 224);
    const dataUrl = canvas.toDataURL('image/jpeg');

    try {
      pendingRequest = true;
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });
      const data = await resp.json();
      if (data.error) {
        labelSpan.textContent = 'Error';
        confSpan.textContent = data.error;
      } else {
        labelSpan.textContent = data.label;
        confSpan.textContent = (data.confidence * 100).toFixed(1) + '%';
      }
    } catch (e) {
      console.error('Predict error', e);
      labelSpan.textContent = 'Error';
      confSpan.textContent = e.message;
    } finally {
      pendingRequest = false;
    }
  }

  startBtn.addEventListener('click', async () => {
    await startCamera();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    const interval = parseInt(intervalInput.value) || 250;
    timer = setInterval(sendFrame, interval);
  });

  stopBtn.addEventListener('click', () => {
    if (timer) clearInterval(timer);
    timer = null;
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
  });
});
