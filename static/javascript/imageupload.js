document.addEventListener("DOMContentLoaded", function () {
    // Retrieve and display form data from localStorage
    const storedData = localStorage.getItem("patientData");

    if (storedData) {
        const patientData = JSON.parse(storedData);

        // Populate the patient-info-container with the retrieved data
        document.getElementById("age").textContent = patientData.age;
        document.getElementById("gender").textContent = patientData.gender;
        document.getElementById("tobacco").textContent = patientData.tobacco;
        document.getElementById("alcohol").textContent = patientData.alcohol;
        document.getElementById("status").textContent = patientData.status;
        document.getElementById("hpv").textContent = patientData.hpv;
        document.getElementById("continent").textContent = patientData.continent;
    }

    // Image upload functionality
    const imageUpload = document.getElementById("imageUpload");
    const chosenImage = document.getElementById("chosenImage");
    const imageContainer = document.getElementById("imageContainer");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");

    imageUpload.addEventListener("change", function () {
        const file = imageUpload.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                chosenImage.src = e.target.result;
                uploadPlaceholder.style.display = "none";
                imageContainer.style.display = "block";
            };

            reader.readAsDataURL(file);
        }
    });
});

// Function to change uploaded image
function changeImage() {
    document.getElementById("imageUpload").click();
}