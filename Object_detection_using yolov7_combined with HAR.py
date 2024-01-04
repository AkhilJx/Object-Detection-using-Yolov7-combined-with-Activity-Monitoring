# 1. Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
from YOLOv7 import YOLOv7


# 2. Initialize YOLOv7 object detector
model_path = ".\\models\\yolov7_384x640.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)


# 3. Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# 4.function for mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# 5.function for drawing landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections


# 6.function for drawing styled landmarks
def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )


# 7. function for calculating the necessary angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# 8. function for calculating the necessary distances
def calculate_dist(a,b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)
    return dist


# 9. function for standardising the angle values
def n1(x):
    return ((x-0)/(180-0))


# 10. function for standardising the keypoint values
def n2(x):
    return ((x-0)/(33-0))


# 11. function to return necessary parameters
def get_coordinates(l):
    landmarks = l

    # Calculate Coordinates for left parameters
    lshoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
    lhip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
    lankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,
              landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
    lknee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,
             landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
    lfootidx = [landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    #     lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    #     lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Calculate Coordinates for right parameters
    rshoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                 landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
    rhip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
    rankle = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x,
              landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]
    rknee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x,
             landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
    rfootidx = [landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    #     relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    #     rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # Calculate required PARAMETERS
    rhangle = calculate_angle(rshoulder, rhip, rknee)
    lhangle = calculate_angle(lshoulder, lhip, lknee)
    rkangle = calculate_angle(rankle, rknee, rhip)
    lkangle = calculate_angle(lankle, lknee, lhip)
    lfangle = calculate_angle(lknee, lankle, lfootidx)
    rfangle = calculate_angle(rknee, rankle, rfootidx)
    ldist = calculate_dist(lhip, lankle)
    rdist = calculate_dist(rhip, rankle)
    #     langle = calculate_angle(lshoulder, lelbow, lwrist)
    #     rangle = calculate_angle(rshoulder, relbow, rwrist)
    #     lsangle=calculate_angle(lhip,lshoulder,lelbow)
    #     rsangle=calculate_angle(rhip,rshoulder,relbow)
    #     ankdist=calculate_dist(lankle,rankle)
    #     rwdist=calculate_dist(rhip,rwrist)
    #     lwdist=calculate_dist(lhip,lwrist)

    return rhangle, lhangle, rkangle, lkangle, lfangle, rfangle, ldist, rdist


# 12. function for extracting the keypoints
def extract_keypoints(results):
    test=[]
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(141)
    if results.pose_landmarks:
        a,b,c,d,e,f,g,h = get_coordinates(results.pose_landmarks.landmark)
        test=np.array([n1(a),n1(b),n1(c),n1(d),n1(e),n1(f),g,h,n2(len(results.pose_landmarks.landmark))])
        q=np.concatenate([pose,test])
        q=q.flatten()
        return q

    else:
        return pose


# 13. Defining the key parameters
# 13.1 Actions that we try to detect
actions = np.array(['standing', 'sitting', 'kneeling'])

# 13.2 Thirty videos worth of data
no_sequences = 100

# 13.3 Videos are going to be 30 frames in length
sequence_length = 30

# 13.4 Folder start
start_folder = 0


# 14. Define Labels and Features

label_map = {label:num for num, label in enumerate(actions)}

# 15. define the LSTM Neural Network model and the associated libraries

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 15.1 Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,141)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# 16. Load the LSTM Neural Network model
model.load_weights('.\\models\\action2.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


# 17. Function to depict the probability status of the human activity
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    # Define the rectangle parameters
    rect_y = 70  # Y-coordinate for the top of the rectangle
    rect_height = len(actions) * 40 + 45  # Height of the rectangle based on the number of actions
    rect_color = (255, 255, 255)  # Color of the rectangle

    # Draw the rectangle
    cv2.rectangle(output_frame, (0, rect_y), (150, rect_y + rect_height), rect_color,thickness=-1)

    for num, prob in enumerate(res):
        y_position = 120 + num * 40  # Adjust this value to move the text lower
        cv2.rectangle(output_frame, (10, y_position), (int(prob * 100), y_position + 30), colors[num], -1)
        cv2.putText(output_frame, actions[num], (10, y_position + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (164, 20, 52), 2,
                    cv2.LINE_AA)
        cv2.putText(output_frame, "::Activity::", (0, 100), cv2.FONT_ITALIC, 1, (0, 0, 255), 2,
                    cv2.LINE_4)

    return output_frame


# 18. Test in Real Time

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)

# We need to set resolutions. so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
result = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Update object localizer
        boxes, scores, class_ids = yolov7_detector(image)

        combined_img, label = yolov7_detector.draw_detections(image)

        # 1. Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #             print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # 3. Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:]

            # Viz probabilities
            combined_img = prob_viz(res, actions, combined_img, colors)


        cv2.rectangle(combined_img, (0, 0), (640, 40), (245, 117, 16), -1)
        if len(sentence) == 0:

            cv2.putText(combined_img, 'The person is not doing ', (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_img, 'any activity', (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:

            cv2.putText(combined_img, 'The person is ', (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_img, str(sentence) , (240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # saving the output
        result.write(combined_img)

        cv2.putText(combined_img, 'Press "q" to quit', (200, 70),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # Show to screen in fullscreen
        cv2.namedWindow('Output', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Output', combined_img)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()