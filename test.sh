#!/bin/bash

#fmpeg -i "$FILE" -vf "select=not(mod(n\,3))" -vsync vfr data/frames/%03d.jpg

source $HOME/miniconda2/bin/activate magenta

#conda activate magenta

  # --frame_skips=0 \
  # --image_size=512 \
  # --style_image_size=256 \

set -e

VIDEOPATH=$1
STYLE=$2

python arbitrary_image_stylization_with_weights.py \
  --logtostderr \
  --video_path=$VIDEOPATH \
  --style_images_path=$HOME/Documents/styles/$STYLE.jpg \
  --interpolation_weight=1 \
  --style_image_size=512 \
  --image_size=1080 \
  --frame_skip=0

#conda deactivate
