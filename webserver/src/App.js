import React, { useState, useEffect } from 'react';
import './App.css';
// import io from 'socket.io-client';

function App() {
    const [file, setFile] = useState(null);
    const [segmentedImage, setSegmentedImage] = useState(null);
    const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
    const [originalVideoUrl, setOriginalVideoUrl] = useState(null);
    const [status, setStatus] = useState('');
    const [showModal, setShowModal] = useState(false);
    const [canPlayOriginal, setCanPlayOriginal] = useState(false);

    // useEffect(() => {
    //     const socket = io('http://localhost:5000');
    //     socket.on('video_processed', (data) => {
    //         setProcessedVideoUrl(data.url);
    //         setStatus('Video processing complete.');
    //     });
    //     return () => socket.disconnect();
    // }, []);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            setFile(file);
            setOriginalVideoUrl(URL.createObjectURL(file));
        }
    };

    const handlePredict = async () => {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('http://localhost:5000/api/segment', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseJson = await response.json();
            if (responseJson.mask) {
                const maskBase64 = responseJson.mask;
                setSegmentedImage(maskBase64);
                setStatus('Prediction complete.');
            }
        } catch (error) {
            setStatus('Failed to connect to the API. Please ensure the Flask API is running.');
            setShowModal(true);
        }

        setCanPlayOriginal(true);
    };

    const clearTempDirectory = async () => {
        try {
            const response = await fetch('http://localhost:5000/api/clear', {
                method: 'POST',
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const responseJson = await response.json();
            if (responseJson.message) {
                setStatus('Temporary directory cleared.');
            }
        } catch (error) {
            // Ignore error since it should not affect the user experience.
            // I'll delete the files next time the user clears while connected 
            // to the API.
        }
    }

    const clearFiles = () => {
        setFile(null);
        setSegmentedImage(null);
        setOriginalVideoUrl(null);
        setProcessedVideoUrl(null);
        clearTempDirectory();
    };

    return (
        <div className="App">
            <h1>Semantic Segmentation</h1>
            {!file && (
                <input
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileUpload}
                    className="upload-btn"
                />
            )}
            {file && (
                <div>
                    <h3>Uploaded</h3>
                    {file.type.startsWith('image/') ? (
                        <img src={URL.createObjectURL(file)} alt="Uploaded file for segmentation" />
                    ) : file.type.startsWith('video/') ? (
                        <video controls src={originalVideoUrl} alt="Original video" autoPlay={canPlayOriginal} />
                    ) : (
                        <p>Unsupported file type.</p>
                    )}
                </div>
            )}
            {segmentedImage && (
                <div>
                    <h3>Predicted</h3>
                    <img src={segmentedImage} alt="Segmented file with semantic labels" />
                </div>
            )}
            {processedVideoUrl && (
                <div>
                    <h3>Processed Video</h3>
                    <video controls src={processedVideoUrl} alt="Processed video" />
                </div>
            )}
            <div className="button-container">
                {file && (
                    <button onClick={clearFiles} className="clear-btn">
                        Clear File
                    </button>
                )}
                {file && (
                    <button onClick={handlePredict} className="predict-btn">
                        Predict
                    </button>
                )}
            </div>
            {showModal && (
                <div className="Modal">
                    <button
                        className="Modal-close"
                        onClick={() => setShowModal(false)}
                    >
                        X
                    </button>
                    <h2 className="error-message-title">Error</h2>
                    <p className="error-message">{status}</p>
                </div>
            )}
        </div>
    );
}

export default App;
