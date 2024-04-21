/**
 * Reference: https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py
 */
import React from 'react'

const Legend = () => {
    const labels = [
        { name: 'unlabeled', color: [0, 0, 0] },
        { name: 'ego vehicle', color: [0, 0, 0] },
        { name: 'rectification border', color: [0, 0, 0] },
        { name: 'out of roi', color: [0, 0, 0] },
        { name: 'static', color: [0, 0, 0] },
        { name: 'dynamic', color: [111, 74, 0] },
        { name: 'ground', color: [81, 0, 81] },
        { name: 'road', color: [128, 64, 128] },
        { name: 'sidewalk', color: [244, 35, 232] },
        { name: 'parking', color: [250, 170, 160] },
        { name: 'rail track', color: [230, 150, 140] },
        { name: 'building', color: [70, 70, 70] },
        { name: 'wall', color: [102, 102, 156] },
        { name: 'fence', color: [190, 153, 153] },
        { name: 'guard rail', color: [180, 165, 180] },
        { name: 'bridge', color: [150, 100, 100] },
        { name: 'tunnel', color: [150, 120, 90] },
        { name: 'pole', color: [153, 153, 153] },
        { name: 'polegroup', color: [153, 153, 153] },
        { name: 'traffic light', color: [250, 170, 30] },
        { name: 'traffic sign', color: [220, 220, 0] },
        { name: 'vegetation', color: [107, 142, 35] },
        { name: 'terrain', color: [152, 251, 152] },
        { name: 'sky', color: [70, 130, 180] },
        { name: 'person', color: [220, 20, 60] },
        { name: 'rider', color: [255, 0, 0] },
        { name: 'car', color: [0, 0, 142] },
        { name: 'truck', color: [0, 0, 70] },
        { name: 'bus', color: [0, 60, 100] },
        { name: 'caravan', color: [0, 0, 90] },
        { name: 'trailer', color: [0, 0, 110] },
        { name: 'train', color: [0, 80, 100] },
        { name: 'motorcycle', color: [0, 0, 230] },
        { name: 'bicycle', color: [119, 11, 32] },
        { name: 'license plate', color: [0, 0, 142] },
    ]

    return (
        <div
            style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
            }}
        >
            <h3
                style={{
                    color: 'gray',
                    fontSize: '14px',
                    marginBottom: '10px',
                }}
            >
                Legend
            </h3>
            {labels.map((label, index) => (
                <div
                    key={index}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        marginBottom: '10px',
                    }}
                >
                    <div
                        style={{
                            width: '10px',
                            height: '10px',
                            borderRadius: '50%',
                            backgroundColor: `rgb(${label.color[0]}, ${label.color[1]}, ${label.color[2]})`,
                            marginRight: '5px',
                        }}
                    ></div>
                    <div style={{ color: 'gray', fontSize: '12px' }}>
                        {label.name}
                    </div>
                </div>
            ))}
        </div>
    )
}

export default Legend
