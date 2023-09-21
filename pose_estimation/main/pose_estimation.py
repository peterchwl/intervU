# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run pose classification and pose estimation."""
import argparse
import logging
import sys
import time

import cv2
from cv2 import cvtColor, COLOR_BGR2GRAY, circle, putText
from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet
import utils
import math
from deepface import DeepFace

import numpy as np
import module as m
import time


def run(estimation_model: str, tracker_type: str, classification_model: str,
        label_file: str, camera_id: int, width: int, height: int):
  """Continuously run inference on images acquired from the camera.

  Args:
    estimation_model: Name of the TFLite pose estimation model.
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
    classification_model: Name of the TFLite pose classification model.
      (Optional)
    label_file: Path to the label file for the pose classification model. Class
      names are listed one name per line, in the same order as in the
      classification model output. See an example in the yoga_labels.txt file.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """

  # Notify users that tracker is only enabled for MoveNet MultiPose model.
  if tracker_type and (estimation_model != 'movenet_multipose'):
    logging.warning(
        'No tracker will be used as tracker can only be enabled for '
        'MoveNet MultiPose model.')

  # Initialize the pose estimator selected.
  if estimation_model in ['movenet_lightning', 'movenet_thunder']:
    pose_detector = Movenet(estimation_model)
  elif estimation_model == 'posenet':
    pose_detector = Posenet(estimation_model)
  elif estimation_model == 'movenet_multipose':
    pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
  else:
    sys.exit('ERROR: Model is not supported.')

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  depth_counter = 0
  ln_counter = 0
  rn_counter = 0
  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  classification_results_to_show = 3
  fps_avg_frame_count = 10
  keypoint_detection_threshold_for_classifier = 0.1
  classifier = None

  # Initialize the classification model
  if classification_model:
    classifier = Classifier(classification_model, label_file)
    classification_results_to_show = min(classification_results_to_show,
                                         len(classifier.pose_class_names))

  picframe = 0
  orig_arr = []
  all_emotions = []
  eye_track = []
  sumR = sumC = sumL = sumD = sumU = 0

  # gets the first 5 frames as a benchmark 
  while picframe <= 5:
      # get original image
      r, f = cap.read()
      if estimation_model == 'movenet_multipose':
        # Run pose estimation using a MultiPose model.
        list_persons = pose_detector.detect(f)
      else:
        # Run pose estimation using a SinglePose model, and wrap the result in an
        # array.
        list_persons = [pose_detector.detect(f)]
      img, arr = utils.visualize(f, list_persons)
      orig_arr = arr
      # save original  image
      cv2.imwrite("./sitting/original.jpg", f)
      picframe = picframe + 1

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, imageorig = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1

    if estimation_model == 'movenet_multipose':
      # Run pose estimation using a MultiPose model.
      list_persons = pose_detector.detect(imageorig)
    else:
      # Run pose estimation using a SinglePose model, and wrap the result in an
      # array.
      list_persons = [pose_detector.detect(imageorig)]

    # Draw keypoints and edges on input image
    image = utils.visualize(imageorig, list_persons)

    if classifier:
      # Check if all keypoints are detected before running the classifier.
      # If there's a keypoint below the threshold, show an error.
      person = list_persons[0]
      min_score = min([keypoint.score for keypoint in person.keypoints])
      if min_score < keypoint_detection_threshold_for_classifier:
        error_text = 'Some keypoints are not detected.'
        text_location = (left_margin, 2 * row_size)
        cv2.putText(image[0], error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
        error_text = 'Make sure the person is fully visible in the camera.'
        text_location = (left_margin, 3 * row_size)
        cv2.putText(image[0], error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
      else:
        # Run pose classification
        prob_list = classifier.classify_pose(person)

        # Show classification results on the image
        for i in range(classification_results_to_show):
          class_name = prob_list[i].label
          probability = round(prob_list[i].score, 2)
          result_text = class_name + ' (' + str(probability) + ')'
          text_location = (left_margin, (i + 2) * row_size)
          cv2.putText(image[0], result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(image[0], fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # print("orig arr: " + str(orig_arr))
    # print("curr arr: " + str(image[1]))
    
    diff_shoulders = abs(orig_arr[0] - image[1][0])
    diff_ln = orig_arr[1] - image[1][1]
    diff_rn = orig_arr[2] - image[1][2]
    
    if diff_shoulders > 10:
      # print("depth failed")
      depth_counter = depth_counter + 1
    elif diff_ln > 10:
      # print("right failed")
      ln_counter = ln_counter + 1
    elif diff_rn > 10:
      # print("left failed")
      rn_counter = rn_counter + 1

    # converting frame into Gry image.
    grayFrame = cvtColor(imageorig, COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width/5), 50)
    # calling the face detector funciton
    imageface, face = m.faceDetector(imageorig, grayFrame)
    if face is not None:
      # calling landmarks detector funciton.
      imageface, PointList = m.faceLandmakDetector(imageface, grayFrame, face, False)
      # print(PointList)

      RightEyePoint = PointList[36:42]
      LeftEyePoint = PointList[42:48]
      leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
      rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

      mask, pos, color, right, center, left, down, up= m.EyeTracking(imageface, grayFrame, RightEyePoint)
      maskleft, leftPos, leftColor, lRight, lCenter, lLeft, lDown, lUp = m.EyeTracking(
          imageface, grayFrame, LeftEyePoint)
      sumR += right + lRight
      sumC += center + lCenter
      sumL += left + lLeft
      sumD += down + lDown
      sumU += up + lUp
    eye_track = [sumR, sumC, sumL, sumD, sumU]
    
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow(estimation_model, image[0])

    result = DeepFace.analyze(image[0], actions=['emotion'], enforce_detection=False, silent=True)
    all_emotions.append(result[0]['emotion']['happy'])
  
  cap.release()
  cv2.destroyAllWindows()
  return [counter, depth_counter, ln_counter, rn_counter], all_emotions, eye_track


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of estimation model.',
      required=False,
      default='movenet_lightning')
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  parser.add_argument(
      '--classifier', help='Name of classification model.', required=False)
  parser.add_argument(
      '--label_file',
      help='Label file for classification.',
      required=False,
      default='labels.txt')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  output = run(args.model, args.tracker, args.classifier, args.label_file,
      int(args.cameraId), args.frameWidth, args.frameHeight)

  err_counters = output[0]
  
  depth = err_counters[1] / err_counters[0]
  left_tilt = err_counters[2] / err_counters[0]
  right_tilt = err_counters[3] / err_counters[0]

  all_emotions = output[1]
  
  aggregated_emotion = sum(all_emotions) / len(all_emotions)
  
  eye_track = output[2]


  print("--------------------------------------------")
  print("Running Tests:")
  print()
  depth = True
  left = True
  right = True
  
  posture = False
  facials = False
  eye_contact = False
  
  print("Test 1: Posture")
  if depth > 0.5:
    print("- depth leaned in/out (fail)")
    depth = False
    
  if left_tilt > 0.3:
    print("- tilted left (relative to screen) (fail)")
    left = False
    
  if right_tilt > 0.3:
    print("- tilted right (relative to screen) (fail)")
    right = False
    
  if depth and left and right:
    print("- posture checked (pass)")
    posture = True

  print("Test 2: Facials")
  if aggregated_emotion < 3:
    print("- not happy/positive (fail)")
  else:
    print("- expressing happiness (pass)")
    facials = True
 
  print("Test 3: Eye Contact")
  if eye_track[1] > eye_track[0] and eye_track[1] > eye_track[2] and eye_track[1] > eye_track[3] and eye_track[1] > eye_track[4]:
    print("- eyes centered (pass)")
    eye_contact = True
  else:
    print("- eyes not centered (fail)")

  print()
  if posture and facials and eye_contact:
    print("> Test overall: Pass <")
  else:
    print("> Test overall: Fail <")

if __name__ == '__main__':
  main()
