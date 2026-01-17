document.addEventListener('DOMContentLoaded', () => {
    console.log('Phone Cough Classifier frontend loaded.');

    const checkHealthBtn = document.getElementById('check-health-btn');
    const statusDiv = document.getElementById('status');

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
    renderAdminPanel();
});

function renderAdminPanel() {
    const tableBody = document.querySelector('#referral-table tbody');
    if (!tableBody) return;

    // Spoof Data for Patient Referrals
    const referralData = [
        { phone: '+1 (555) 123-4567', doctor: 'Dr. Sarah Smith', date: '2026-01-18', status: 'success' },
        { phone: '+1 (555) 987-6543', doctor: 'Dr. Raj Patel', date: '2026-01-18', status: 'pending' },
        { phone: '+1 (555) 456-7890', doctor: 'Dr. Emily Chen', date: '2026-01-17', status: 'failure' },
        { phone: '+1 (555) 222-3333', doctor: 'Dr. Alan Turing', date: '2026-01-16', status: 'success' },
        { phone: '+1 (555) 888-9999', doctor: 'Dr. Grace Hopper', date: '2026-01-15', status: 'success' },
    ];

    referralData.forEach(item => {
        const row = document.createElement('tr');
        
        let statusLabel = '';
        let statusClass = '';

        switch(item.status) {
            case 'success':
                statusLabel = 'Checked In';
                statusClass = 'status-success';
                break;
            case 'failure':
                statusLabel = 'Missed';
                statusClass = 'status-failure';
                break;
            case 'pending':
                statusLabel = 'Pending';
                statusClass = 'status-pending';
                break;
            default:
                statusLabel = 'Unknown';
                statusClass = 'status-pending';
        }

        row.innerHTML = `
            <td>${item.phone}</td>
            <td>${item.doctor}</td>
            <td>${item.date}</td>
            <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
        `;
        tableBody.appendChild(row);
    });
}