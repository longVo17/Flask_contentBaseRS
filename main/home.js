let users = [];
let currentUser = null;
let currentLocation = 'all';
let popularJobs = []; // Store popular jobs to reuse

// Load user data and popular jobs on page load
document.addEventListener('DOMContentLoaded', () => {
  loadUserData();
  loadPopularJobs(); // Load jobs once
  updateSectionVisibility(); // Ensure correct section is shown initially
});

// Load user data
function loadUserData() {
  fetch('USER_DATA_FINAL.csv')
    .then(response => response.text())
    .then(csvText => {
      Papa.parse(csvText, {
        header: true,
        complete: function(results) {
          users = results.data.filter(user => user['UserID'] && user['UserID'].trim());
          populateUserSelect();
        }
      });
    })
    .catch(error => {
      console.error('Error loading user data:', error);
    });
}

// Populate user select dropdown
function populateUserSelect() {
  const select = document.getElementById('userSelect');
  select.innerHTML = '<option value="">-- Select Profile --</option>';

  users.forEach((user, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = user['User Name'] || `User ${index + 1}`;
    select.appendChild(option);
  });

  select.addEventListener('change', function() {
    const userIndex = this.value;
    if (userIndex !== '') {
      currentUser = users[userIndex];
      displayUserProfile(currentUser);
    } else {
      document.getElementById('userProfile').style.display = 'none';
      document.getElementById('jobList').innerHTML = '';
    }
  });
}

function displayUserProfile(user) {
  const profileDiv = document.getElementById('userProfile');
  const contentDiv = document.getElementById('profileContent');

  let html = '';
  Object.keys(user).forEach(key => {
    if (user[key] && user[key].trim() && !key.toLowerCase().includes('id')) {
      if (key === 'URL User') {
        // Hiển thị URL User dưới dạng liên kết
        html += `<div class="profile-item">
          <strong>${key}:</strong> <a href="${user[key]}" target="_blank" rel="noopener">${user[key]}</a>
        </div>`;
      } else {
        html += `<div class="profile-item">
          <strong>${key}:</strong> ${user[key]}
        </div>`;
      }
    }
  });

  contentDiv.innerHTML = html;
  profileDiv.style.display = 'block';
}
// Show CV creator section
function showCVCreator() {
  document.getElementById('welcomeSection').style.display = 'none';
  document.getElementById('cvCreator').classList.add('active');
  document.getElementById('jobFinder').classList.remove('active');
  updateSectionVisibility();
}

// Show job finder section
function showJobFinder() {
  document.getElementById('welcomeSection').style.display = 'none';
  document.getElementById('jobFinder').classList.add('active');
  document.getElementById('cvCreator').classList.remove('active');
  updateSectionVisibility();
}

// Back to welcome section
function backToWelcome() {
  document.getElementById('welcomeSection').style.display = 'block';
  document.getElementById('cvCreator').classList.remove('active');
  document.getElementById('jobFinder').classList.remove('active');
  document.getElementById('jobList').innerHTML = '';
  updateSectionVisibility();
}

// Save CV and get recommendations
function saveCV() {
  const cvData = {
    name: document.getElementById('userName').value,
    email: document.getElementById('userEmail').value,
    phone: document.getElementById('userPhone').value,
    desiredJob: document.getElementById('desiredJob').value,
    industry: document.getElementById('industry').value,
    workplace: document.getElementById('workplaceDesired').value,
    desiredSalary: document.getElementById('desiredSalary').value,
    gender: document.getElementById('userGender').value,
    marriage: document.getElementById('userMarriage').value,
    age: document.getElementById('userAge').value,
    target: document.getElementById('userTarget').value,
    skills: document.getElementById('skills').value,
    degree: document.getElementById('userDegree').value,
    workExperience: document.getElementById('workExperience').value
  };

  if (!cvData.name || !cvData.desiredJob || !cvData.skills) {
    alert('Please fill in all required fields (Full Name, Desired Position, Skills)!');
    return;
  }

  // Show loading effect
  const saveButton = document.querySelector('#cvCreator .btn-primary');
  saveButton.disabled = true;
  saveButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving CV...';

  fetch('http://localhost:5000/save_cv', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(cvData)
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    if (data.error) {
      alert(`Error: ${data.error}`);
    } else {
      alert('CV has been saved successfully!');
      // Update users array with new user
      users.push(data.user_info);
      // Update user select dropdown
      populateUserSelect();
      // Select the new user
      document.getElementById('userSelect').value = users.length - 1;
      currentUser = data.user_info;
      displayUserProfile(currentUser);
      // Display recommended jobs
      const jobListDiv = document.getElementById('jobList');
      jobListDiv.innerHTML = '';
      displayJobs(data.recommended_jobs);
      // Switch to job finder to show recommendations
      showJobFinder();
    }
  })
  .catch(error => {
    console.error('Error saving CV:', error);
    alert(`Error saving CV: ${error.message}`);
  })
  .finally(() => {
    // Reset button after request completes
    saveButton.disabled = false;
    saveButton.innerHTML = '<i class="fas fa-save"></i> Save CV';
  });

  // Clear form
  document.querySelectorAll('#cvCreator input, #cvCreator textarea, #cvCreator select').forEach(field => {
    field.value = '';
  });
}

// Find jobs based on user profile
function findJobs() {
  const userIndex = document.getElementById('userSelect').value;
  const jobCount = document.getElementById('jobCount').value;

  if (!userIndex) {
    alert('Please select a profile to find matching jobs!');
    return;
  }

  const jobListDiv = document.getElementById('jobList');
  jobListDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Searching for matching jobs...</div>';

  fetch('http://localhost:5000/recommend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_index: parseInt(userIndex),
      top_n: parseInt(jobCount)
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    if (data.error) {
      jobListDiv.innerHTML = `<div class="loading">❌ ${data.error}</div>`;
    } else {
      displayJobs(data.recommended_jobs);
    }
  })
  .catch(error => {
    console.error('Error:', error);
    jobListDiv.innerHTML = `<div class="loading">❌ Error loading jobs: ${error.message}</div>`;
  });
}

// Display jobs
function displayJobs(jobs) {
  const jobListDiv = document.getElementById('jobList');

  if (!jobs || jobs.length === 0) {
    jobListDiv.innerHTML = '<div class="loading">No matching jobs found.</div>';
    return;
  }

  let html = '';
  jobs.forEach((job, index) => {
    const similarity = Math.round((job['Similarity Score'] || 0) * 100);
    const badges = getBadges(similarity, index);

    html += `
      <div class="job-card" onclick="showJobDetail('${job.JobID}')">
        <div class="job-header">
          <div class="job-badges">${badges}</div>
        </div>
        <div class="job-title">${job['Job Title'] || 'No title'}</div>
        <div class="job-company">${job['Name Company'] || 'Unknown company'}</div>
        <div class="job-details">
          <div class="job-detail-item">
            <i class="fas fa-chart-line"></i>
            <span>${job['Career Level'] || 'Not specified'}</span>
          </div>
          <div class="job-detail-item">
            <i class="fas fa-dollar-sign"></i>
            <span>${job['Salary'] || 'Negotiable'}</span>
          </div>
          <div class="job-detail-item">
            <i class="fas fa-map-marker-alt"></i>
            <span>${job['Job Address'] || 'Not specified'}</span>
          </div>
        </div>
        <div class="similarity-bar">
          <div class="similarity-label">
            <span>Match Score</span>
            <span>${similarity}%</span>
          </div>
          <div class="similarity-progress">
            <div class="similarity-fill" style="width: ${similarity}%"></div>
          </div>
        </div>
      </div>
    `;
  });

  jobListDiv.innerHTML = html;

  // Animate similarity bars
  setTimeout(() => {
    document.querySelectorAll('.similarity-fill').forEach(bar => {
      bar.style.width = bar.style.width;
    });
  }, 100);
}

// Get badges based on similarity
function getBadges(similarity, index) {
  let badges = '';

  if (similarity >= 80) {
    badges += '<span class="badge badge-top">TOP</span>';
  }

  if (index < 3) {
    badges += '<span class="badge badge-hot">HOT</span>';
  }

  if (similarity >= 70) {
    badges += '<span class="badge badge-new">MATCH</span>';
  }

  return badges;
}

// Show job detail
function showJobDetail(jobId) {
  window.location.href = `jobDetail.html?job_id=${jobId}`;
}

// Switch location filter
function switchLocation(location) {
  currentLocation = location;

  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  event.target.classList.add('active');

  // Filter jobs by location (implement based on requirements)
}

// Load and display popular jobs
function loadPopularJobs() {
  const jobListDiv = document.getElementById('popularJobList');
  jobListDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Loading popular jobs...</div>';

  fetch('http://localhost:5000/popular_jobs')
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(jobs => {
      if (!jobs || jobs.length === 0) {
        jobListDiv.innerHTML = '<div class="loading">No popular jobs available.</div>';
        return;
      }
      popularJobs = jobs.slice(0, 100); // Store jobs globally
      displayPopularJobs(popularJobs); // Display initially
    })
    .catch(error => {
      console.error('Error loading popular jobs:', error);
      jobListDiv.innerHTML = `<div class="loading">❌ Error loading jobs: ${error.message}</div>`;
    });
}

// Display popular jobs
function displayPopularJobs(jobs) {
  const jobListDiv = document.getElementById('popularJobList');
  let html = '';
  jobs.forEach(job => {
    html += `
      <div class="job-card" onclick="showJobDetail('${job.JobID}')">
        <div class="job-header">
          ${job['Job Title'] === 'TOP' ? '<div class="job-badges"><span class="badge badge-top">TOP</span></div>' : ''}
        </div>
        <div class="job-title">${job['Job Title'] || 'No title'}</div>
        <div class="job-company">${job['Name Company'] || 'Unknown company'}</div>
        <div class="job-details">
          <div class="job-detail-item">
            <i class="fas fa-dollar-sign"></i>
            <span>${job['Salary'] || 'Negotiable'}</span>
          </div>
          <div class="job-detail-item">
            <i class="fas fa-map-marker-alt"></i>
            <span>${job['Job Address'] || 'No information'}</span>
          </div>
        </div>
        <div class="job-actions">
          <i class="far fa-heart heart-icon"></i>
        </div>
      </div>
    `;
  });
  jobListDiv.innerHTML = html;
}

// Update section visibility
function updateSectionVisibility() {
  const welcomeSection = document.getElementById('welcomeSection');
  const cvCreator = document.getElementById('cvCreator');
  const jobFinder = document.getElementById('jobFinder');
  const popularJobsSection = document.getElementById('popularJobsSection');

  if (cvCreator.classList.contains('active') || jobFinder.classList.contains('active')) {
    popularJobsSection.style.display = 'none';
  } else {
    popularJobsSection.style.display = 'block';
    if (popularJobs.length > 0) {
      displayPopularJobs(popularJobs); // Re-render if data is loaded
    }
  }
}