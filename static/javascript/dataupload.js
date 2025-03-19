document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");

    form.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent actual form submission

        // Extract form data
        const formData = {
            age: document.querySelector("[name='age']").value,
            gender: document.querySelector("[name='gender']").value,
            tobacco: document.querySelector("[name='tobacco']").value,
            alcohol: document.querySelector("[name='alcohol']").value,
            status: document.querySelector("[name='status']").value,
            hpv: document.querySelector("[name='hpv']").value,
            continent: document.querySelector("[name='continent']").value
        };

        // Store form data in localStorage
        localStorage.setItem("patientData", JSON.stringify(formData));

        // Redirect to image upload page
        window.location.href = "imageupload.html";
    });
});
