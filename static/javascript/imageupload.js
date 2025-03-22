function previewImage() {
    const input = document.getElementById('imageUpload');
    const imageContainer = document.getElementById('imageContainer');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const chosenImage = document.getElementById('chosenImage');

    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            chosenImage.src = e.target.result;
            imageContainer.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function changeImage() {
    document.getElementById('imageUpload').click();
}