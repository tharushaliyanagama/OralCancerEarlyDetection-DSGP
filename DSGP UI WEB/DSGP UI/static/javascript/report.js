// Simulate receiving data (image, probability, description).
window.onload = function() {
    const imageSrc = 'heatmap_Example.jpeg';
    const probability = 45;
    const description = "This heatmap highlights the areas that were most important for the AI’s classification. The red and yellow regions indicate zones of high attention, meaning the AI found potential signs of abnormality in these areas. These patterns could correspond to lesions, discoloration, or irregular textures commonly associated with oral cancer. Green or blue areas had less influence, indicating the AI considered them less relevant to the prediction.";

    document.getElementById('uploadedImage').src = imageSrc;
    document.getElementById('probabilityText').innerText = `Probability: ${probability}%`;
    document.getElementById('descriptionText').innerText = description;

    const cautionLine = document.getElementById('cautionLine');
    if (probability > 75) {
        cautionLine.style.backgroundColor = 'red';
    } else if (probability > 40) {
        cautionLine.style.backgroundColor = 'orange';
    } else {
        cautionLine.style.backgroundColor = 'green';
    }
};