import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [segmentedImage, setSegmentedImage] = useState(null);
    const [segmentedVideo, setSegmentedVideo] = useState(null);
    const [originalVideoUrl, setOriginalVideoUrl] = useState(null);
    const [status, setStatus] = useState('');
    const [showModal, setShowModal] = useState(false);
    const [canPlayOriginal, setCanPlayOriginal] = useState(false);
    const [isVideo, setIsVideo] = useState(false);
    const [streamUrl, setStreamUrl] = useState(null);
    const [isCleared, setIsCleared] = useState(false);

    useEffect(() => {
        // Assuming the Flask server is running on localhost:5000
        setStreamUrl('http://localhost:5000/video_feed');
    }, []);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            setFile(file);
            setOriginalVideoUrl(URL.createObjectURL(file));
            setIsVideo(file.type.startsWith('video/'));
        }
    };

    const fetchMjpegUrl = async () => {
        try {
            const response = await fetch('http://localhost:5000/video_feed');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const streamUrl = response.url;
            setStreamUrl(streamUrl);
        } catch (error) {
            console.log("Error here!")
            setStatus('Failed to connect to the MJPEG stream. Please ensure the Flask server is running.');
            setShowModal(true);
        }
    };

    const handlePredict = async () => {
        if (isVideo) {
            await fetchMjpegUrl();
        }

        try {
            const formData = new FormData();
            formData.append('file', file);

            const fileType = file.type.startsWith('image/') ? 'image' : file.type.startsWith('video/') ? 'video' : 'unknown';
            formData.append('fileType', fileType);

            const response = await fetch('http://localhost:5000/api/segment', {
                method: 'POST',
                body: formData,
            });

            if (isVideo) {
                setSegmentedVideo(true);
                setCanPlayOriginal(true);
            }

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
            if (!isVideo || !isCleared) {
                setStatus('Failed to connect to the API. Please ensure the Flask API is running.');
                setShowModal(true);
            }
            setIsCleared(false);
        }
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
        setSegmentedVideo(null);
        setOriginalVideoUrl(null);
        clearTempDirectory();
        setIsCleared(true);
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
            {segmentedVideo && (
                <div>
                    <h3>Predicted</h3>
                    <img src={streamUrl} alt="Streamed video frames" />
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
