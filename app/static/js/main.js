/**
 * Swasth Saathi - Main Application Logic
 * Handles Recording, API Interaction, and UI Updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Swasth Saathi initialized.');
    
    // Initialize components
    const recorder = new VoiceRecorder();
    const map = new HealthMap();
    const ui = new UIController();

    // Bind recording events
    ui.bindRecordingEvents(
        () => recorder.start(),
        () => recorder.stop(),
        () => recorder.analyze()
    );

    // Recorder callbacks
    recorder.onRecordingStart = () => ui.setRecordingState(true);
    recorder.onRecordingStop = (audioUrl) => ui.setRecordingState(false, audioUrl);
    recorder.onAnalysisStart = () => ui.setAnalyzingState(true);
    recorder.onAnalysisComplete = (result) => ui.showResult(result);
    recorder.onAnalysisError = (error) => ui.showError(error);
    recorder.onTimeUpdate = (seconds) => ui.updateTimer(seconds);

    // Initialize Map
    map.init();
});

/**
 * Handles Audio Recording logic using MediaRecorder API.
 */
class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.timerInterval = null;
        this.secondsRecorded = 0;
        
        // Callbacks
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onAnalysisStart = null;
        this.onAnalysisComplete = null;
        this.onAnalysisError = null;
        this.onTimeUpdate = null;
    }

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Determine supported mime type
            let mimeType = 'audio/webm';
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mimeType = 'audio/mp4';
            } else if (MediaRecorder.isTypeSupported('audio/wav')) {
                mimeType = 'audio/wav';
            }
            console.log(`Using MIME type: ${mimeType}`);

            this.mediaRecorder = new MediaRecorder(stream, { mimeType });
            this.audioChunks = [];
            this.secondsRecorded = 0;

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this._finalizeRecording(mimeType);
            };

            this.mediaRecorder.start(1000); // Chunk every second
            this._startTimer();
            
            if (this.onRecordingStart) this.onRecordingStart();

        } catch (err) {
            console.error('Microphone access error:', err);
            alert('Could not access microphone. Please ensure permissions are granted.');
        }
    }

    stop() {
        console.log("VoiceRecorder: Stop requested");
        this._stopTimer(); // Stop timer immediately

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            console.log("VoiceRecorder: Stopping media recorder...");
            this.mediaRecorder.stop();
            
            // Stop all tracks immediately to release mic
            if (this.mediaRecorder.stream) {
                this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        } else {
            console.warn("VoiceRecorder: Recorder was already inactive or null.");
            // Force UI update even if recorder was weird state
            if (this.onRecordingStop) this.onRecordingStop(null); 
        }
    }

    _finalizeRecording(mimeType) {
        console.log("VoiceRecorder: Finalizing recording blob...");
        this.audioBlob = new Blob(this.audioChunks, { type: mimeType });
        const audioUrl = URL.createObjectURL(this.audioBlob);
        
        // Determine extension based on mimeType for filename
        let ext = 'webm';
        if (mimeType.includes('mp4')) ext = 'mp4';
        if (mimeType.includes('wav')) ext = 'wav';
        this.fileExtension = ext;

        // Notify UI that the FILE is ready (UI might already be in processing state)
        if (this.onRecordingStop) this.onRecordingStop(audioUrl);
    }

    async analyze() {
        if (!this.audioBlob) return;

        if (this.onAnalysisStart) this.onAnalysisStart();

        const formData = new FormData();
        formData.append('audio_file', this.audioBlob, `recording.${this.fileExtension}`);

        try {
            const response = await fetch('/test/classify', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            if (this.onAnalysisComplete) this.onAnalysisComplete(result);

        } catch (error) {
            console.error('Analysis failed:', error);
            if (this.onAnalysisError) this.onAnalysisError(error.message);
        }
    }

    _startTimer() {
        this.timerInterval = setInterval(() => {
            this.secondsRecorded++;
            if (this.onTimeUpdate) this.onTimeUpdate(this.secondsRecorded);
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }
}

/**
 * Handles UI updates and event binding.
 */
class UIController {
    constructor() {
        this.recordBtn = document.getElementById('record-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.audioPreview = document.getElementById('audio-preview');
        this.statusText = document.getElementById('status-text');
        this.visualizerBars = document.getElementById('visualizer-bars');
        this.avatarContainer = document.getElementById('avatar-container');
        this.micIcon = document.getElementById('mic-icon');
        
        // Result Section
        this.resultSection = document.getElementById('result-section');
        this.resClass = document.getElementById('res-classification');
        this.resSeverityBar = document.getElementById('severity-bar');
        this.resConfidence = document.getElementById('res-confidence');
        this.resRecommendation = document.getElementById('res-recommendation');
        this.doctorConnect = document.getElementById('doctor-connect');
    }

    bindRecordingEvents(startCallback, stopCallback, analyzeCallback) {
        // Main Button
        if (this.recordBtn) {
            this.recordBtn.onclick = (e) => {
                e.preventDefault();
                console.log("UI: Record button clicked");
                startCallback();
            };
        }

        // Avatar Click (also acts as a button)
        if (this.avatarContainer) {
            this.avatarContainer.onclick = (e) => {
                e.preventDefault();
                if (this.stopBtn && !this.stopBtn.classList.contains('hidden')) {
                    console.log("UI: Avatar clicked (Stop)");
                    this.stopBtn.click(); // Trigger stop logic
                } else if (this.recordBtn && !this.recordBtn.classList.contains('hidden')) {
                    console.log("UI: Avatar clicked (Start)");
                    startCallback();
                }
            };
        }
        
        if (this.stopBtn) {
            this.stopBtn.onclick = (e) => {
                e.preventDefault();
                console.log("UI: Stop button clicked");
                
                // Immediate UI Feedback
                this.stopBtn.disabled = true; // Prevent double clicks
                this.stopBtn.classList.add('opacity-50', 'cursor-wait');
                this.statusText.textContent = 'Processing audio...';
                
                stopCallback();
            };
        }
        
        if (this.analyzeBtn) {
            this.analyzeBtn.onclick = (e) => {
                e.preventDefault();
                console.log("UI: Analyze button clicked");
                analyzeCallback();
            };
        }
    }

    setRecordingState(isRecording, audioUrl = null) {
        if (isRecording) {
            this.recordBtn.classList.add('hidden');
            this.stopBtn.classList.remove('hidden');
            this.statusText.textContent = 'Listening... (0s)';
            this.statusText.className = "mt-4 text-sm font-bold text-accent min-h-[20px] animate-pulse";
            
            // Visualizer
            if (this.visualizerBars) this.visualizerBars.style.opacity = '1';
            if (this.micIcon) this.micIcon.style.opacity = '0';
            if (this.avatarContainer) this.avatarContainer.classList.add('scale-110');
            
            // Reset previous results
            this.resultSection.classList.add('hidden', 'opacity-0', 'translate-y-4');
            this.audioPreview.classList.add('hidden');
            this.analyzeBtn.classList.add('hidden');

        } else {
            // Stopped
            this.recordBtn.classList.remove('hidden');
            this.stopBtn.classList.add('hidden');
            
            // Reset Stop Button State (in case it was disabled during processing)
            this.stopBtn.disabled = false;
            this.stopBtn.classList.remove('opacity-50', 'cursor-wait');

            this.statusText.textContent = 'Recording captured. Ready to analyze.';
            this.statusText.className = "mt-4 text-sm font-medium text-slate-500 min-h-[20px]";

            // Reset Visualizer
            if (this.visualizerBars) this.visualizerBars.style.opacity = '0';
            if (this.micIcon) this.micIcon.style.opacity = '1';
            if (this.avatarContainer) this.avatarContainer.classList.remove('scale-110');

            // Show Audio & Analyze Button
            if (audioUrl) {
                this.audioPreview.src = audioUrl;
                this.audioPreview.classList.remove('hidden');
                
                this.analyzeBtn.disabled = false;
                this.analyzeBtn.classList.remove('hidden', 'opacity-50', 'cursor-not-allowed');
                this.analyzeBtn.innerHTML = 'Analyze Now';
                this.analyzeBtn.classList.add('animate-bounce');
            }
        }
    }

    updateTimer(seconds) {
        this.statusText.textContent = `Listening... (${seconds}s)`;
    }

    setAnalyzingState(isAnalyzing) {
        if (isAnalyzing) {
            this.analyzeBtn.disabled = true;
            this.analyzeBtn.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Analyzing...`;
            this.analyzeBtn.classList.remove('animate-bounce');
            
            // Show result section placeholder
            this.resultSection.classList.remove('hidden');
            setTimeout(() => {
                this.resultSection.classList.remove('opacity-0', 'translate-y-4');
            }, 10);
            this.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            this.analyzeBtn.disabled = false;
            this.analyzeBtn.textContent = 'Analyze Again';
        }
    }

    showResult(data) {
        this.setAnalyzingState(false);
        
        // Update Text
        this.resClass.textContent = data.classification || "Unknown";
        this.resRecommendation.textContent = data.recommendation ? `"${data.recommendation}"` : "No recommendation available.";
        
        // Update Confidence
        const conf = data.confidence ? (data.confidence * 100).toFixed(1) : "0.0";
        this.resConfidence.textContent = `${conf}%`;

        // Update Severity Bar & Doctor Connect
        const severity = (data.severity || 'low').toLowerCase();
        let width = '10%';
        let color = 'bg-green-500';
        let showDoctor = false;

        if (['urgent', 'high', 'severe'].includes(severity)) {
            width = '90%';
            color = 'bg-red-500';
            showDoctor = true;
        } else if (['moderate', 'medium'].includes(severity)) {
            width = '50%';
            color = 'bg-yellow-500';
        }

        this.resSeverityBar.className = `h-full ${color} transition-all duration-1000 ease-out rounded-full relative`;
        this.resSeverityBar.style.width = '0%';
        setTimeout(() => {
            this.resSeverityBar.style.width = width;
        }, 100);

        if (showDoctor) {
            this.doctorConnect.classList.remove('hidden');
        } else {
            this.doctorConnect.classList.add('hidden');
        }
    }

    showError(msg) {
        this.setAnalyzingState(false);
        this.resClass.textContent = "Error";
        this.resRecommendation.textContent = msg;
    }
}

/**
 * Handles Map Visualization
 */
class HealthMap {
    init() {
        const mapDiv = document.getElementById('map');
        if (!mapDiv || mapDiv._leaflet_id) return;

        const map = L.map('map').setView([20.5937, 78.9629], 5);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        this._loadData(map);
    }

    async _loadData(map) {
        try {
            const response = await fetch('/admin/heatmap');
            let data = await response.json();

            if (!data || data.length === 0) {
                // Fallback data
                data = [
                    { city: "New Delhi", lat: 28.6139, lng: 77.2090, count: 18, high_risk: 8 },
                    { city: "Mumbai", lat: 19.0760, lng: 72.8777, count: 25, high_risk: 12 },
                    { city: "Bangalore", lat: 12.9716, lng: 77.5946, count: 8, high_risk: 1 }
                ];
            }

            data.forEach(point => {
                const riskRatio = point.count > 0 ? (point.high_risk / point.count) : 0;
                let color = '#0D9488'; // Teal
                if (riskRatio > 0.5) color = '#E11D48'; // Rose
                else if (riskRatio > 0.2) color = '#F59E0B'; // Amber

                const radius = Math.max(10, Math.min(point.count * 2, 50)); 

                L.circleMarker([point.lat, point.lng], {
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.6,
                    radius: radius,
                    weight: 1
                })
                .addTo(map)
                .bindPopup(`<b>${point.city}</b><br>Cases: ${point.count}`);
            });

        } catch (error) {
            console.error("Error loading heatmap:", error);
        }
    }
}
