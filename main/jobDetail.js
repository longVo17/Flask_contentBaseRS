// Get job ID from URL parameters
const urlParams = new URLSearchParams(window.location.search);
const jobId = urlParams.get('job_id');

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    if (!jobId) {
        showError('Error: Job ID not found');
    } else {
        loadJobDetail(jobId);
    }
});

// Function to load job detail from API
function loadJobDetail(jobId) {
    fetch(`http://localhost:5000/job_detail?job_id=${jobId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            renderJobDetail(data);
        })
        .catch(error => {
            console.error('Error fetching job detail:', error);
            showError(`Error loading data: ${error.message}`);
        });
}

// Function to render job detail
function renderJobDetail(job) {
    const jobUrl = job['URL Job'] || '#';

    document.getElementById('jobDetail').innerHTML = `
        <!-- Header Section -->
        <div class="header">
            <div class="job-title">
                ${job['Job Title'] || 'No title'}
                <span class="verified-badge">✓</span>
            </div>
            
            <div class="job-meta">
                <div class="meta-item">
                    <div class="meta-icon salary-icon">₫</div>
                    <div class="meta-content">
                        <h4>Salary</h4>
                        <p>${job['Salary'] || 'Negotiable'}</p>
                    </div>
                </div>
                
                <div class="meta-item">
                    <div class="meta-icon location-icon">📍</div>
                    <div class="meta-content">
                        <h4>Location</h4>
                        <p>${job['Job Address'] || 'No information'}</p>
                    </div>
                </div>
                
                <div class="meta-item">
                    <div class="meta-icon experience-icon">💼</div>
                    <div class="meta-content">
                        <h4>Experience</h4>
                        <p>${job['Years of Experience'] || 'Not required'}</p>
                    </div>
                </div>
                
                <div class="meta-item">
                    <div class="meta-icon deadline-icon">⏰</div>
                    <div class="meta-content">
                        <h4>Application Deadline</h4>
                        <p>${job['Submission Deadline'] || 'Not specified'}</p>
                    </div>
                </div>
            </div>
            
            <div class="action-buttons">
                <a href="${jobUrl}" target="_blank" class="btn-apply">
                    ✉️ Apply Now
                </a>
                <button class="btn-save" onclick="saveJob()">
                    ♡ Save Job
                </button>
            </div>
        </div>

        <!-- Company Info Section -->
        <div class="content-section">
            <h2 class="section-title">Company Information</h2>
            <div class="company-info">
                <div>
                    <div class="company-logo">🏢</div>
                    <div class="company-details">
                        <h3>${job['Name Company'] || 'No company name'}</h3>
                        <p style="color: #6c757d; margin-bottom: 15px;">${job['Company Overview'] || 'No company description'}</p>
                        <p><strong>Address:</strong> ${job['Company Address'] || 'No address'}</p>
                    </div>
                </div>
                
                <div class="company-stats">
                    <div class="stat-item">
                        <div class="stat-icon">👥</div>
                        <div>
                            <div style="font-size: 12px; color: #6c757d;">Size</div>
                            <div style="font-weight: 600;">${job['Company Size'] || 'No information'}</div>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-icon">🏭</div>
                        <div>
                            <div style="font-size: 12px; color: #6c757d;">Industry</div>
                            <div style="font-weight: 600;">${job['Industry'] || 'No information'}</div>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-icon">⚧</div>
                        <div>
                            <div style="font-size: 12px; color: #6c757d;">Gender</div>
                            <div style="font-weight: 600;">${job['Gender'] || 'Not required'}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Job Description Section -->
        <div class="content-section">
            <h2 class="section-title">Job Details</h2>
            <div class="job-content">
                ${formatJobDescription(job)}
            </div>
        </div>

        <!-- Job Requirements Section -->
        <div class="content-section">
            <h2 class="section-title">Job Requirements</h2>
            <div class="job-content">
                ${formatJobRequirements(job)}
            </div>
        </div>

        <!-- Benefits Section -->
        <div class="content-section">
            <h2 class="section-title">Benefits</h2>
            <div class="job-content">
                ${formatBenefits(job)}
            </div>
        </div>

        <!-- Additional Info Section -->
        <div class="content-section">
            <h2 class="section-title">Additional Information</h2>
            <div class="job-content">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div>
                        <h4>Job Type</h4>
                        <p>${job['Job Type'] || 'No information'}</p>
                    </div>
                    <div>
                        <h4>Level</h4>
                        <p>${job['Career Level'] || 'No information'}</p>
                    </div>
                    <div>
                        <h4>Number of Openings</h4>
                        <p>${job['Number Cadidate'] || 'No information'}</p>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Function to format job description
function formatJobDescription(job) {
    const description = job['Job Description'] || 'No job description';
    return formatText(description);
}

// Function to format job requirements
function formatJobRequirements(job) {
    const requirements = job['Job Requirements'] || 'No specific requirements';
    return formatText(requirements);
}

// Function to format benefits
function formatBenefits(job) {
    const benefits = job['Benefits'] || 'No information on benefits';
    return formatText(benefits);
}

// Function to format text (convert line breaks to HTML)
function formatText(text) {
    if (!text || text === 'null' || text === 'undefined') {
        return '<p>No information</p>';
    }

    // Convert line breaks to HTML paragraphs
    const paragraphs = text.split('\n').filter(p => p.trim() !== '');
    if (paragraphs.length === 0) {
        return '<p>No information</p>';
    }

    return paragraphs.map(p => `<p>${p.trim()}</p>`).join('');
}

// Function to show error message
function showError(message) {
    document.getElementById('jobDetail').innerHTML = `
        <div class="content-section">
            <h2 style="color: #F44336; text-align: center;">${message}</h2>
        </div>
    `;
}

// Function to save job (placeholder)
function saveJob() {
    // You can implement save job functionality here
    alert('Save job feature will be updated soon!');
}

// Function to handle apply button click (if needed for tracking)
function applyJob(jobUrl) {
    // You can add tracking or other functionality here before redirecting
    window.open(jobUrl, '_blank');
}