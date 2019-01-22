# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates stylized images with different strengths of a stylization.

For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
See run_interpolation_with_identity.sh for example usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
import sys
import time
import subprocess
import librosa

from math import *

import cv2

import numpy as np
import tensorflow as tf

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils

slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_string('checkpoint', "model.ckpt", 'Path to the model checkpoint.')
flags.DEFINE_string('style_images_path', "/home/edeetee/Documents/styles/mosaic.jpg", 'Paths to the style images'
                    'for evaluation.')
flags.DEFINE_string('content_images_paths', None, 'Paths to the content images'
                    'for evaluation.')

flags.DEFINE_string('video_path', "/home/edeetee/Videos/SSURFACES.mkv", 'Paths to the content images'
                    'for evaluation.')
flags.DEFINE_integer('frame_skips', 0, 'for each frame processed, skip x frames')

flags.DEFINE_string('output_dir', "output", 'Output directory.')
flags.DEFINE_integer('image_size', 512, 'Image size.')
flags.DEFINE_boolean('content_square_crop', False, 'Wheather to center crop'
                     'the content image to be a square or not.')
flags.DEFINE_integer('style_image_size', 512, 'Style image size.')
flags.DEFINE_boolean('style_square_crop', True, 'Wheather to center crop'
                     'the style image to be a square or not.')

flags.DEFINE_integer('maximum_styles_to_evaluate', 1024, 'Maximum number of'
                     'styles to evaluate.')
flags.DEFINE_float('interpolation_weight', 0.8, 'weight of identity'
                    'for interpolation between the parameters of the identity'
                    'transform and the style parameters of the style image. The'
                    'larger the weight is the strength of stylization is more.'
                    'Weight of 1.0 means the normal style transfer and weight'
                    'of 0.0 means identity transform.')
FLAGS = flags.FLAGS


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MkDir(FLAGS.output_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Defines place holder for the style image.
    style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    if FLAGS.style_square_crop:
        style_img_preprocessed = image_utils.center_crop_resize_image(
            style_img_ph, FLAGS.style_image_size)
    else:
        style_img_preprocessed = image_utils.resize_image(style_img_ph,
            FLAGS.style_image_size)


    #video input
    capture = cv2.VideoCapture(FLAGS.video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(FLAGS.video_path)[:-4]

    in_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    in_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    FLAGS.image_size = int(min(in_width, in_height, FLAGS.image_size))

    # Defines place holder for the content image.
    content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    if FLAGS.content_square_crop:
        content_img_preprocessed = image_utils.center_crop_resize_image(
            content_img_ph, FLAGS.image_size)
    else:
      content_img_preprocessed = image_utils.resize_image(
          content_img_ph, FLAGS.image_size)

    # Defines the model.
    stylized_images, _, _, bottleneck_feat = build_model.build_model(
        content_img_preprocessed,
        style_img_preprocessed,
        trainable=False,
        is_training=False,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=False)

    if tf.gfile.IsDirectory(FLAGS.checkpoint):
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
    else:
        checkpoint = FLAGS.checkpoint
        tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))

    init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                             slim.get_variables_to_restore())
    sess.run([tf.local_variables_initializer()])
    init_fn(sess)

    # Gets the list of the input style images.
    style_img_path = tf.gfile.Glob(FLAGS.style_images_path)[0]
    print("\nstyling using " + os.path.basename(style_img_path) + " at " + str(FLAGS.image_size) + "p")
    style = os.path.basename(style_img_path)[:-4]
    style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]

    # Computes bottleneck features of the style prediction network for the
    # given style image.
    style_params = sess.run(
        bottleneck_feat, feed_dict={style_img_ph: style_image_np})

    #video output
    width = int(FLAGS.image_size*in_width/in_height)
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    out_fps = fps/(1+FLAGS.frame_skips)
    out_file=os.path.join(FLAGS.output_dir, video_name + "_" + style + "_" + str(FLAGS.image_size) + '.avi')
    out = cv2.VideoWriter(out_file, codec, out_fps, (width,FLAGS.image_size), True)

    #audio input
    cmd = "ffmpeg -y -loglevel quiet -i {} -ab 160k -ac 2 -ar 44100 -vn {}.wav".format(FLAGS.video_path, video_name)
    subprocess.call(cmd, shell=True)
    y, sr = librosa.load(video_name+".wav")
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=10)
    feature_split = 1
    # rms = librosa.feature.rmse(y=y, frame_length=int(sr/out_fps/feature_split))[0]
    bins_per_octave=4
    hop_length=int(sr/out_fps)
    n_bins=bins_per_octave*5
    # cqt = np.abs(librosa.core.cqt(y, sr=sr, fmin=30, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length))
    cqt = np.abs(librosa.core.stft(y, hop_length=hop_length))
    cqt_sr = sr/hop_length

    output_files = []
    hasFrame = capture.isOpened()
    i = 0
    start = time.time()
    maxWeight = 1
    lastWeight = 0
    while(True):
        frame_start = time.time()
        # skip frames
        for skip in range(FLAGS.frame_skips):
            capture.grab()

        hasFrame, frame = capture.read()
        
        if not hasFrame:
            break

        inp = cv2.resize(frame, (FLAGS.image_size, FLAGS.image_size))
        content_img_np = inp[:, :, [2, 1, 0]]
        content_img_name = video_name + "_" + str(i)

        # for content_i, content_img_path in content_enum:
        # if video:
        #   content_img_np = video[content_i]
        #   content_img_name = str(content_i)
        # else:
        #   content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :3]
        #   content_img_name = os.path.basename(content_img_path)[:-4]

        # Saves preprocessed content image.
        # inp_img_croped_resized_np = sess.run(
        #     content_img_preprocessed, feed_dict={
        #         content_img_ph: content_img_np
        #     })
        # image_utils.save_np_image(inp_img_croped_resized_np,
        #                           os.path.join(FLAGS.output_dir,
        #                                       '%s.jpg' % (content_img_name)))

        # Computes bottleneck features of the style prediction network for the
        # identity transform.
        identity_params = sess.run(bottleneck_feat, feed_dict={style_img_ph: content_img_np})

        duration = time.time()-start

        # while beats[0] < duration:
        #     beats = beats[1:]
        # weight = max(0, min(1, abs(beats[0]-duration)))

        weight = 0

        # print(cqt.shape[0])
        bin_start = int(cqt.shape[0]*0.001)
        bins = int(cqt.shape[0]*0.25)
        for bin_i in range(bin_start, bin_start+bins):
            cur = cqt[bin_i, int(cqt_sr*i/out_fps)]
            weight += cur

        weight = weight/bins
        # weight = min(1, min(1, weight/bins)*FLAGS.interpolation_weight)

        # weight = min(1, cqt[])
        # weight = 1 - FLAGS.interpolation_weight * (1+sin(i/7*pi))/2

        # print(weight)
        maxWeight=max(maxWeight, weight)
        weight /= maxWeight
        weight *= weight
        weight = max(weight, lastWeight-0.1)

        lastWeight = weight

        stylized_image_res = sess.run(
            stylized_images,
            feed_dict={
                bottleneck_feat:
                    identity_params * (1 - weight) + style_params * weight,
                content_img_ph:
                    content_img_np
            })

        # output_filename = '%s_stylized_%s.jpg' % (content_img_name, style)
        # output_file = os.path.join(FLAGS.output_dir, output_filename)

        # Saves stylized image.
        # image_utils.save_np_image(stylized_image_res, output_file)
        # output_files += output_file

        #writes image to video output
        sqr_output_frame = np.uint8(stylized_image_res * 255)[0][:,:,[2,1,0]]
        out.write(cv2.resize(sqr_output_frame, (width, FLAGS.image_size)))
        
        tf.logging.info('Stylized %s with weight %.2f at %.1f fps' % 
            (content_img_name, weight, 1/(time.time()-frame_start)))

        # print("Outputted " + '%s_stylized_%s.jpg' %
        #     (content_img_name, style))

        i += 1
    
    out.release()
    capture.release()
    cmd="ffmpeg -i {} -i {}.wav -c:v copy -shortest -map 0:v:0 -map 1:a:0 temp.avi".format(out_file, video_name)
    print(cmd)
    subprocess.call(cmd, shell=True)
    subprocess.call("mv -f temp.avi {}".format(out_file), shell=True)
    subprocess.call("rm {}.wav".format(video_name), shell=True)

    print( "Average fps: " + str(i/(time.time()-start)) )
    return 0

if __name__ == '__main__':
  tf.app.run(main)
