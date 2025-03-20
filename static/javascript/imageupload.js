document.addEventListener("DOMContentLoaded", async function () {
    // Fetch patient data from backend
    try {
        const response = await fetch("http://localhost:3000/get-patient-data");
        const patientData = await response.json();

        if (Object.keys(patientData).length > 0) {
            document.getElementById("age").textContent = patientData.age;
            document.getElementById("gender").textContent = patientData.gender;
            document.getElementById("tobacco").textContent = patientData.tobacco;
            document.getElementById("alcohol").textContent = patientData.alcohol;
            document.getElementById("status").textContent = patientData.status;
            document.getElementById("hpv").textContent = patientData.hpv;
            document.getElementById("continent").textContent = patientData.continent;
        }
    } catch (error) {
        console.error("Error fetching patient data:", error);
    }

    // Image upload functionality
    const imageUpload = document.getElementById("imageUpload");
    const chosenImage = document.getElementById("chosenImage");
    const imageContainer = document.getElementById("imageContainer");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");

    imageUpload.addEventListener("change", async function () {
        const file = imageUpload.files[0];

        if (file) {
            const formData = new FormData();
            formData.append("image", file);

            try {
                const uploadResponse = await fetch("http://localhost:3000/upload-image", {
                    method: "POST",
                    body: formData
                });

                const result = await uploadResponse.json();
                if (result.imageUrl) {
                    chosenImage.src = `http://localhost:3000${result.imageUrl}`;
                    uploadPlaceholder.style.display = "none";
                    imageContainer.style.display = "block";
                }
            } catch (error) {
                console.error("Error uploading image:", error);
            }
        }
    });
});

// Function to change uploaded image
function changeImage() {
    document.getElementById("imageUpload").click();
}
