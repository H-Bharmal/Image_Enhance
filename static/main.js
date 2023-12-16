document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const resultContainer = document.getElementById('result-container');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(uploadForm);

        fetch('/process_image', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Response data:', data);

            if (data.error) {
                console.error('Server Error:', data.error);
            } else {
                const resultImage = new Image();
                resultImage.src = `data:image/png;base64,${data.result}`;
                resultContainer.innerHTML = '';
                resultContainer.appendChild(resultImage);
            }
        })
        .catch(error => {
            console.error('Fetch error:', error.message);
        });
    });
});


// document.addEventListener('DOMContentLoaded', function() {
//     const uploadForm = document.getElementById('upload-form');
//     const resultContainer = document.getElementById('result-container');

//     uploadForm.addEventListener('submit', function(event) {
//         event.preventDefault();

//         const formData = new FormData(uploadForm);

//         fetch('/process_image', {
//             method: 'POST',
//             body: formData,
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log('Response data:', data);  // Add this line for debugging

//             if (data.error) {
//                 console.error('Error:', data.error);
//             } else {
//                 const resultImage = new Image();
//                 resultImage.src = `data:image/png;base64,${data.result}`;
//                 resultContainer.innerHTML = '';
//                 resultContainer.appendChild(resultImage);
//             }
//         })
//         .catch(error => {
//             console.error('Fetch error:', error);
//         });
//     });
// });
