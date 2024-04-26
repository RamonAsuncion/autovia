#!/bin/bash

frontend() {
    cd webserver
    npm start
}

backend() {
    python backend.py --model_weights_path ./weights/model-bnet-2.pth
}

# run frontend and backend in parallel
frontend & backend