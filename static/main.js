// Ensure the script runs after DOM is ready and provide an interactive Camera/Upload UI
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext ? canvas.getContext('2d') : null;
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const labelSpan = document.getElementById('label');
  const confSpan = document.getElementById('conf');
  const intervalInput = document.getElementById('interval');
  const fileInput = document.getElementById('fileInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const previewImg = document.getElementById('preview');
  const pendingIndicator = document.getElementById('pendingIndicator');
  const overlay = document.getElementById('overlay');
  const historyList = document.getElementById('historyList');
  const dropArea = document.getElementById('dropArea');
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabs = document.querySelectorAll('.tab');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');

  let stream = null;
  let timer = null;
  let pendingRequest = false;
  let predictionHistory = [];

  function renderHistory() {
    if (!historyList) return;
    historyList.innerHTML = '';
    predictionHistory.forEach(item => {
      const li = document.createElement('li');
      const t = new Date(item.time).toLocaleTimeString();
      li.textContent = `${t} — ${item.label} (${(item.conf*100).toFixed(1)}%)`;
      historyList.appendChild(li);
    });
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Webcam not supported in this browser.');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (video) video.srcObject = stream;
      await (video ? video.play() : Promise.resolve());
    } catch (e) {
      alert('Could not access webcam: ' + (e.message || e));
    }
  }

  async function sendFrame() {
    if (!video || !ctx || (video.readyState < 2)) return;
    if (pendingRequest) return;
    const side = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, 224, 224);
    const dataUrl = canvas.toDataURL('image/jpeg');

    try {
      pendingRequest = true;
      if (pendingIndicator) pendingIndicator.style.display = 'inline';
      if (overlay) overlay.textContent = '...';
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });
      const data = await resp.json();
      if (data.error) {
        labelSpan.textContent = 'Error';
        confSpan.textContent = data.error;
        if (overlay) overlay.textContent = 'Error';
      } else {
        labelSpan.textContent = data.label;
        confSpan.textContent = (data.confidence * 100).toFixed(1) + '%';
        if (overlay) overlay.textContent = `${data.label} ${ (data.confidence*100).toFixed(1)}%`;
        predictionHistory.unshift({ label: data.label, conf: data.confidence, time: Date.now() });
        if (predictionHistory.length > 10) predictionHistory.pop();
        renderHistory();
      }
    } catch (e) {
      console.error('Predict error', e);
      labelSpan.textContent = 'Error';
      confSpan.textContent = e.message || String(e);
      if (overlay) overlay.textContent = 'Error';
    } finally {
      pendingRequest = false;
      if (pendingIndicator) pendingIndicator.style.display = 'none';
    }
  }

  async function predictFromDataUrl(dataUrl) {
    if (pendingRequest) return;
    try {
      pendingRequest = true;
      if (previewImg) { previewImg.src = dataUrl; previewImg.style.display = 'block'; }
      if (pendingIndicator) pendingIndicator.style.display = 'inline';
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
        predictionHistory.unshift({ label: data.label, conf: data.confidence, time: Date.now() });
        if (predictionHistory.length > 10) predictionHistory.pop();
        renderHistory();
      }
    } catch (e) {
      console.error('Predict error', e);
      labelSpan.textContent = 'Error';
      confSpan.textContent = e.message || String(e);
    } finally {
      pendingRequest = false;
      if (pendingIndicator) pendingIndicator.style.display = 'none';
    }
  }

  // Camera controls
  if (startBtn) {
    startBtn.addEventListener('click', async () => {
      await startCamera();
      if (startBtn) startBtn.disabled = true;
      if (stopBtn) stopBtn.disabled = false;
      const interval = parseInt(intervalInput.value) || 250;
      timer = setInterval(sendFrame, interval);
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      if (timer) clearInterval(timer);
      timer = null;
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
      if (overlay) overlay.textContent = '-';
    });
  }

  // File upload flow
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) { alert('Please choose a file first'); return; }
      const reader = new FileReader();
      reader.onload = (ev) => predictFromDataUrl(ev.target.result);
      reader.readAsDataURL(file);
    });

    fileInput.addEventListener('change', () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      if (previewImg) { previewImg.src = url; previewImg.style.display = 'block'; }
      // switch to upload tab
      switchToTab('upload');
      if (stream && stopBtn) stopBtn.click();
    });
  }

  // drag-and-drop
  if (dropArea) {
    ['dragenter', 'dragover'].forEach(evt => dropArea.addEventListener(evt, (e) => { e.preventDefault(); dropArea.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(evt => dropArea.addEventListener(evt, (e) => { e.preventDefault(); dropArea.classList.remove('dragover'); }));
    dropArea.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const file = dt.files && dt.files[0];
      if (!file) return;
      fileInput.files = dt.files;
      const reader = new FileReader();
      reader.onload = (ev) => predictFromDataUrl(ev.target.result);
      reader.readAsDataURL(file);
      switchToTab('upload');
      if (stream && stopBtn) stopBtn.click();
    });
  }

  function switchToTab(name) {
    tabs.forEach(t => t.classList.remove('active'));
    tabButtons.forEach(b => b.classList.remove('active'));
    const tab = document.getElementById('tab-' + name);
    if (tab) tab.classList.add('active');
    const btn = document.querySelector(`.tab-btn[data-tab="${name}"]`);
    if (btn) btn.classList.add('active');
    // stricter behavior: when switching to upload, ensure camera is stopped and start button disabled
    if (name === 'upload') {
      if (stream && stopBtn) stopBtn.click();
      if (startBtn) startBtn.disabled = true;
    } else if (name === 'camera') {
      // allow starting camera manually
      if (startBtn) startBtn.disabled = false;
      if (previewImg) { previewImg.style.display = 'none'; }
    }
  }

  tabButtons.forEach(btn => btn.addEventListener('click', () => {
    const name = btn.getAttribute('data-tab');
    if (name === 'camera') {
      if (previewImg) previewImg.style.display = 'none';
    } else if (name === 'upload') {
      if (stream && stopBtn) stopBtn.click();
    }
    switchToTab(name);
  }));

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      predictionHistory = [];
      renderHistory();
      // clear UI
      if (previewImg) { previewImg.src = ''; previewImg.style.display = 'none'; }
      if (overlay) overlay.textContent = '-';
      if (labelSpan) labelSpan.textContent = '-';
      if (confSpan) confSpan.textContent = '-';
    });
  }

  if (pendingIndicator) pendingIndicator.style.display = 'none';
  renderHistory();
});
