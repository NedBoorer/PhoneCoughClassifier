document.addEventListener('DOMContentLoaded', () => {
    console.log('Swasth Saathi frontend loaded.');

    const checkHealthBtn = document.getElementById('check-health-btn'); // Hidden but kept for logic if needed
    
    // --- Navigation & Language Logic ---
    setupLanguageSwitcher();

    // --- Admin/Map Logic ---
    initMap();

    // --- Recording & Analysis Logic ---
    setupRecordingFlow();
});

// ==========================================
// Language Switcher (Mock Implementation)
// ==========================================
const TRANSLATIONS = {
    'en': {
        title: "Namaste! <br><span class='text-primary bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary'>How is your health today?</span>",
        subtitle: "I am your personal health companion. I can listen to your cough and help you understand your health.",
        btn: "Check Your Cough",
        status: "Tap to start recording (5-10 seconds)",
        analyzing: "Analyzing Cough Pattern..."
    },
    'hi': {
        title: "नमस्ते! <br><span class='text-primary bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary'>आज आपका स्वास्थ्य कैसा है?</span>",
        subtitle: "मैं आपका निजी स्वास्थ्य साथी हूँ। मैं आपकी खांसी सुनकर आपके स्वास्थ्य को समझने में मदद कर सकता हूँ।",
        btn: "अपनी खांसी जांचें",
        status: "रिकॉर्डिंग शुरू करने के लिए टैप करें (5-10 सेकंड)",
        analyzing: "खांसी के पैटर्न का विश्लेषण किया जा रहा है..."
    }
};

function setupLanguageSwitcher() {
    const btnEn = document.getElementById('lang-en');
    const btnHi = document.getElementById('lang-hi');
    const title = document.getElementById('hero-title');
    const subtitle = document.getElementById('hero-subtitle');
    const btnText = document.getElementById('btn-record-text');
    const statusText = document.getElementById('status-text');

    function setLang(lang) {
        // Toggle Active State
        if (lang === 'en') {
            btnEn.className = "text-sm font-medium text-slate-900 bg-white/50 hover:bg-white px-3 py-1 rounded-full transition-colors shadow-sm ring-1 ring-slate-200";
            btnHi.className = "text-sm font-medium text-slate-500 hover:text-slate-900 hover:bg-white/50 px-3 py-1 rounded-full transition-colors";
        } else {
            btnHi.className = "text-sm font-medium text-slate-900 bg-white/50 hover:bg-white px-3 py-1 rounded-full transition-colors shadow-sm ring-1 ring-slate-200";
            btnEn.className = "text-sm font-medium text-slate-500 hover:text-slate-900 hover:bg-white/50 px-3 py-1 rounded-full transition-colors";
        }

        // Update Text
        const t = TRANSLATIONS[lang];
        title.innerHTML = t.title;
        subtitle.textContent = t.subtitle;
        btnText.textContent = t.btn;
        statusText.textContent = t.status;
    }

    btnEn.addEventListener('click', () => setLang('en'));
    btnHi.addEventListener('click', () => setLang('hi'));
}

// ==========================================
// Recording & Analysis Flow
// ==========================================
function setupRecordingFlow() {
    const recordBtn = document.getElementById('record-btn');
    const stopBtn = document.getElementById('stop-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const audioPreview = document.getElementById('audio-preview');
    const statusText = document.getElementById('status-text');
    const visualizerBars = document.getElementById('visualizer-bars');
    const avatarContainer = document.getElementById('avatar-container');
    
    // Result Elements
    const resultSection = document.getElementById('result-section');
    const resClass = document.getElementById('res-classification');
    const resSeverityBar = document.getElementById('severity-bar');
    const resConfidence = document.getElementById('res-confidence');
    const resRecommendation = document.getElementById('res-recommendation');
    const doctorConnect = document.getElementById('doctor-connect');

    let mediaRecorder;
    let audioChunks = [];
    let audioBlob = null;
    let isRecording = false;

    // Avatar Pulse Effect
    function setVisualizerState(active) {
        if (active) {
            visualizerBars.style.opacity = '1';
            avatarContainer.classList.add('scale-110');
        } else {
            visualizerBars.style.opacity = '0';
            avatarContainer.classList.remove('scale-110');
        }
    }

    if (recordBtn && stopBtn && analyzeBtn) {
        // Start Recording
        recordBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPreview.src = audioUrl;
                    audioPreview.classList.remove('hidden');
                    
                    analyzeBtn.disabled = false;
                    analyzeBtn.classList.remove('hidden', 'opacity-50', 'cursor-not-allowed');
                    analyzeBtn.classList.add('animate-bounce'); // Draw attention
                    
                    statusText.textContent = 'Recording captured. Tap "Analyze Now" to see results.';
                    setVisualizerState(false);
                    
                    // Reset UI slightly
                    recordBtn.classList.remove('hidden');
                    stopBtn.classList.add('hidden');
                };

                mediaRecorder.start();
                isRecording = true;
                
                // toggle buttons
                recordBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
                
                statusText.textContent = 'Listening... Please cough clearly.';
                statusText.className = "mt-4 text-sm font-bold text-accent min-h-[20px] animate-pulse";
                setVisualizerState(true);
                
                // Hide previous results if any
                resultSection.classList.add('hidden', 'opacity-0', 'translate-y-4');
                
            } catch (err) {
                console.error('Error accessing microphone:', err);
                alert('Could not access microphone. Please allow permissions.');
            }
        });

        // Stop Recording
        stopBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop()); // Stop stream
                isRecording = false;
                statusText.className = "mt-4 text-sm font-medium text-slate-500 min-h-[20px]";
            }
        });

        // Analyze
        analyzeBtn.addEventListener('click', async () => {
            if (!audioBlob) return;

            // UI Loading State
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Analying...`;
            analyzeBtn.classList.remove('animate-bounce');

            // Show Result Section Placeholder
            resultSection.classList.remove('hidden');
            // Small delay to allow display:block to apply before transition
            setTimeout(() => {
                resultSection.classList.remove('opacity-0', 'translate-y-4');
            }, 10);
            
            // Scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'recording.wav');

            try {
                // Determine API endpoint - use test endpoint for direct classification
                const response = await fetch('/test/classify', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                // Update UI with Result
                updateResultCard(result);

            } catch (error) {
                console.error('Analysis failed:', error);
                resClass.textContent = "Error";
                resRecommendation.textContent = "Could not reach the server. Please try again.";
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze Again';
            }
        });
    }

    function updateResultCard(data) {
        // 1. Classification
        resClass.textContent = data.classification || "Unknown";
        
        // 2. Severity Bar
        // Map severity to percentage: healthy=5%, mild=40%, urgent=90%
        let width = '5%';
        let color = 'bg-green-500';
        let severity = (data.severity || 'low').toLowerCase();

        if (severity === 'urgent' || severity === 'high') {
            width = '90%';
            color = 'bg-red-500';
            doctorConnect.classList.remove('hidden');
        } else if (severity === 'moderate' || severity === 'mild') {
            width = '50%';
            color = 'bg-yellow-500';
            doctorConnect.classList.add('hidden');
        } else {
            // Healthy/Low
            width = '10%';
            color = 'bg-green-500';
            doctorConnect.classList.add('hidden');
        }
        
        // Reset classes and force reflow for animation
        resSeverityBar.className = `h-full ${color} transition-all duration-1000 ease-out rounded-full relative`;
        resSeverityBar.style.width = '0%';
        setTimeout(() => {
            resSeverityBar.style.width = width;
        }, 100);

        // 3. Confidence
        const conf = data.confidence ? (data.confidence * 100).toFixed(1) : "0.0";
        resConfidence.textContent = `${conf}%`;

        // 4. Recommendation
        if (data.recommendation) {
            resRecommendation.textContent = `"${data.recommendation}"`;
        } else {
            resRecommendation.textContent = "No specific recommendation available.";
        }
    }
}

// ==========================================
// Map Logic (Updated Colors)
// ==========================================
async function initMap() {
    const mapDiv = document.getElementById('map');
    if (!mapDiv) return;

    // Center on India
    const map = L.map('map').setView([20.5937, 78.9629], 5);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    try {
        const response = await fetch('/admin/heatmap');
        let data = await response.json();

        // Hack for Demo: Provide Fallback Data
        if (!data || data.length === 0) {
            data = [
                { city: "New Delhi", lat: 28.6139, lng: 77.2090, count: 18, high_risk: 8 },
                { city: "Mumbai", lat: 19.0760, lng: 72.8777, count: 25, high_risk: 12 },
                { city: "Bangalore", lat: 12.9716, lng: 77.5946, count: 8, high_risk: 1 },
                { city: "Chennai", lat: 13.0827, lng: 80.2707, count: 12, high_risk: 3 },
                { city: "Kolkata", lat: 22.5726, lng: 88.3639, count: 15, high_risk: 4 }
            ];
        }

        data.forEach(point => {
            const riskRatio = point.count > 0 ? (point.high_risk / point.count) : 0;
            
            // New Colors: Green (Teal-ish for safe), Red (Rose for danger)
            let color = '#0D9488'; // Teal (Safe)
            if (riskRatio > 0.5) color = '#E11D48'; // Rose (High Risk)
            else if (riskRatio > 0.2) color = '#F59E0B'; // Amber (Moderate)

            const radius = Math.max(10, Math.min(point.count * 2, 50)); 

            const circle = L.circleMarker([point.lat, point.lng], {
                color: color,
                fillColor: color,
                fillOpacity: 0.6,
                radius: radius,
                weight: 1
            }).addTo(map);

            circle.bindPopup(
                `
                <div style="font-family: Inter, sans-serif;">
                    <strong style="color: #1e293b;">${point.city}</strong><br>
                    <span style="font-size: 0.8rem; color: #64748b;">Active Cases: ${point.count}</span><br>
                    <span style="font-size: 0.8rem; color: ${color}; font-weight: bold;">High Risk: ${point.high_risk}</span>
                </div>
            `
            );
        });

    } catch (error) {
        console.error("Error loading heatmap:", error);
    }
}