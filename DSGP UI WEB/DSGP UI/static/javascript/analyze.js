// Optional: Show file name if you passed it from DataUpload.html
const fileName = sessionStorage.getItem('uploadedFileName');
if (fileName) {
    document.getElementById('fileNameDisplay').innerText = `File: ${fileName}`;
}

// Simulate analysis time (replace with actual backend call if needed)
setTimeout(() => {
    // After processing is done, redirect to results page (or wherever you want)
    window.location.href = 'result.html';  // Create a results page later
}, 5000);  // 5 seconds delay