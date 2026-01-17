document.addEventListener('DOMContentLoaded', () => {
    console.log('Phone Cough Classifier frontend loaded.');

    const checkHealthBtn = document.getElementById('check-health-btn');
    const statusDiv = document.getElementById('status');

    // --- Health Check Logic ---
    if (checkHealthBtn) {
        checkHealthBtn.addEventListener('click', async () => {
            statusDiv.style.display = 'block';
            statusDiv.textContent = 'Checking API status...';
            statusDiv.className = '';

            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                statusDiv.textContent = `API Status: ${data.status}. Environment: ${data.environment}`;
                statusDiv.classList.add('success');
                console.log(data);
            } catch (error) {
                console.error('Error checking health:', error);
                statusDiv.textContent = 'Failed to connect to API.';
                statusDiv.classList.add('error');
            }
        });
    }

    // --- Admin Panel Logic ---
    loadReferrals();
    initMap();

    // --- Demo Analysis Logic ---
    const analyzeBtn = document.getElementById('analyze-btn');
    const audioUpload = document.getElementById('audio-upload');
    const analysisResult = document.getElementById('analysis-result');

    if (analyzeBtn && audioUpload) {
        analyzeBtn.addEventListener('click', async () => {
            const file = audioUpload.files[0];
            if (!file) {
                alert('Please select an audio file first.');
                return;
            }

            analysisResult.style.display = 'block';
            analysisResult.textContent = 'Analyzing audio with AI...';
            
            const formData = new FormData();
            formData.append('audio_file', file);

            try {
                const response = await fetch('/test/classify', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                analysisResult.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">Analysis Complete:</div>
                    <div>Classification: <span style="color: var(--primary-color);">${result.classification}</span></div>
                    <div>Severity: <span style="color: ${result.severity === 'urgent' ? 'red' : 'orange'};">${result.severity}</span></div>
                    <div>Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; font-style: italic;">"${result.recommendation}"</div>
                `;
            } catch (error) {
                console.error('Analysis failed:', error);
                analysisResult.textContent = 'Error: Could not analyze audio.';
            }
        });
    }
});

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

        // Hack for Demo: If no data, show sample data
        if (data.length === 0) {
            console.log("No real data, showing demo heatmap data");
            data = [
                { city: "New Delhi", lat: 28.6139, lng: 77.2090, count: 15, high_risk: 5 },
                { city: "Mumbai", lat: 19.0760, lng: 72.8777, count: 25, high_risk: 12 },
                { city: "Bangalore", lat: 12.9716, lng: 77.5946, count: 8, high_risk: 1 }
            ];
        }

        data.forEach(point => {
            // Color based on % of high risk
            const riskRatio = point.count > 0 ? (point.high_risk / point.count) : 0;
            let color = '#22c55e'; // Green
            if (riskRatio > 0.5) color = '#ef4444'; // Red
            else if (riskRatio > 0.2) color = '#eab308'; // Yellow

            // Radius based on total count
            const radius = Math.max(10, Math.min(point.count * 2, 50)); 

            const circle = L.circleMarker([point.lat, point.lng], {
                color: color,
                fillColor: color,
                fillOpacity: 0.6,
                radius: radius
            }).addTo(map);

            circle.bindPopup(
                `
                <b>${point.city}</b><br>
                Total Calls: ${point.count}<br>
                High Risk: ${point.high_risk}
            `
            );
        });

    } catch (error) {
        console.error("Error loading heatmap:", error);
    }
}

async function loadReferrals() {
    const tableBody = document.querySelector('#referral-table tbody');
    if (!tableBody) return;

    try {
        const response = await fetch('/admin/referrals');
        const data = await response.json();
        
        tableBody.innerHTML = ''; // Clear existing

        if (data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No referrals found in database.</td></tr>';
            return;
        }

        data.forEach(item => {
            const row = document.createElement('tr');
            
            const statusLabel = item.verified ? 'Checked In' : 'Pending';
            const statusClass = item.verified ? 'status-success' : 'status-pending';
            
            // Audio player
            const audioHtml = item.recording_url 
                ? `<audio controls src="${item.recording_url}" style="height: 30px; width: 150px;"></audio>` 
                : '<span style="color: #94a3b8; font-size: 0.8rem;">No recording</span>';

            // Verify button
            const actionHtml = item.verified 
                ? '<span style="color: #166534; font-size: 0.8rem;">✓ Verified</span>' 
                : `<button onclick="verifyVisit(${item.id})" style="padding: 0.25rem 0.5rem; font-size: 0.8rem; background-color: #64748b;">Verify Visit</button>`;

            // Health Card Link
            const cardHtml = `<a href="#" onclick="alert('Digital Health Card generation simulated.'); return false;" style="font-size: 0.8rem; margin-left: 0.5rem;">📄 Card</a>`;

            row.innerHTML = `
                <td>${item.phone}</td>
                <td style="text-transform: capitalize;">${item.severity}</td>
                <td>${item.date}</td>
                <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
                <td>${audioHtml}</td>
                <td>${actionHtml} ${cardHtml}</td>
            `;
            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading referrals:', error);
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align:center; color: red;">Error loading data from server.</td></tr>';
    }
}

async function verifyVisit(callId) {
    // 1. Prompt for Doctor's Diagnosis (Ground Truth)
    const diagnosis = prompt("Doctor's Diagnosis (Ground Truth):\n\nExamples: 'Normal', 'Viral Infection', 'Tuberculosis', 'COPD'\n\nPlease enter the actual diagnosis:");
    
    if (diagnosis === null) return; // User cancelled
    if (diagnosis.trim() === "") {
        alert("Diagnosis is required to verify the visit.");
        return;
    }

    const notes = prompt("Additional Notes (Optional):");

    try {
        const response = await fetch(`/admin/verify/${callId}`, { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                diagnosis: diagnosis,
                notes: notes || ""
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            loadReferrals(); // Refresh table
        } else {
            alert('Error: ' + result.message);
        }
    } catch (error) {
        console.error('Error verifying visit:', error);
        alert('Failed to connect to server.');
    }
}