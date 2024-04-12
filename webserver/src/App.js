import React, { useState } from 'react'
import './App.css'

function App() {
    const [image, setImage] = useState(null)
    const [loading, setLoading] = useState(false)

    const handleImageUpload = (event) => {
        setLoading(true)
        const file = event.target.files[0]
        if (file) {
            const reader = new FileReader()
            reader.onloadend = () => {
                setImage(reader.result)
                setLoading(false)
            }
            reader.readAsDataURL(file)
        }
    }

    const clearImage = () => {
        setImage(null)
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
            {image && <img src={image} alt="Uploaded" />}
            {image && (
                <button onClick={clearImage} className="clear-btn">
                    Clear Image
                </button>
            )}
        </div>
    )
}

export default App
