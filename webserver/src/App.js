import React, { useState } from 'react'
import './App.css'

function App() {
    const [image, setImage] = useState(null)
    const [loading, setLoading] = useState(false)
    const [segmentedImage, setSegmentedImage] = useState(null)

    const handleImageUpload = (event) => {
        setLoading(true)
        const file = event.target.files[0]
        if (file) {
            const reader = new FileReader()
            reader.onloadend = () => {
                setImage(reader.result)
                
                const payload = JSON.stringify({'image': reader.result})
                
                // send to inference backend
                fetch('http://localhost:5000/api/segment-image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    mode: "cors",
                    body: payload,
                })
                .then(response => response.json())
                .then(responseJson => {
                    const maskBase64 = responseJson.mask
                    setSegmentedImage(maskBase64)
                })

                setLoading(false)
            }
            reader.readAsDataURL(file)

        }
    }

    const clearImages = () => {
        setImage(null)
        setSegmentedImage(null)
    }

    return (
        <div className="App">
            <h1>Semantic Segmentation</h1>
            <h2>Autonomous Driving</h2>
            {!image && (
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="upload-btn"
                />
            )}
            {loading && <p>Loading...</p>}
            {image && <img src={image} alt="Uploaded Input Image" />}
            {segmentedImage && <img src={segmentedImage} alt="Segmented Image" />}
            {image && segmentedImage && (
                <button onClick={clearImages} className="clear-btn">
                    Clear Image
                </button>
            )}
        </div>
    )
}

export default App
