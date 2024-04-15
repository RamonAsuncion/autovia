#!/bin/bash
source image-env/bin/activate
module switch python/3.10-jvs008
jupyter lab --ip=0.0.0.0 --no-browser