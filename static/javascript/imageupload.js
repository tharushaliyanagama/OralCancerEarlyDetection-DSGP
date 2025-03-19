function previewImage() {
    const fileInput = document.getElementById('imageUpload');
    const imageContainer = document.getElementById('imageContainer');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const chosenImage = document.getElementById('chosenImage');
    const imageDataInput = document.getElementById('imageData');

    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            chosenImage.src = e.target.result;
            imageContainer.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
            imageDataInput.value = e.target.result; // Embed image data in form
        };
        reader.readAsDataURL(file);
    }
}

function changeImage() {
    document.getElementById('imageUpload').click();  // Trigger the file input click
}

function goToAnalyzing() {
   window.location.href = 'analyze.html';
}