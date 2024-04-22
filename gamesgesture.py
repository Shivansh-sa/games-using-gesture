import cv2
import pyautogui
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False,model_complexity=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
def detectPose(image, pose, draw=False, display=False):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),thickness=2, circle_radius=2))
    return output_image, results


def checkLeftRight(image, results, draw=False, display=False):
    horizontal_position = None
    height, width, _ = image.shape
    output_image = image.copy()
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    if (right_x <= width // 2 and left_x <= width // 2):
        horizontal_position = 'Left'
    elif (right_x >= width // 2 and left_x >= width // 2):
        horizontal_position = 'Right'
    elif (right_x >= width // 2 and left_x <= width // 2):
        horizontal_position = 'Center'
    if draw:
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

    return output_image, horizontal_position


def checkJumpCrouch(image, results, MID_Y=250, draw=False):
    height, width, _ = image.shape
    output_image = image.copy()
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    actual_mid_y = abs(right_y + left_y) // 2
    lower_bound = MID_Y - 15
    upper_bound = MID_Y + 50
    if (actual_mid_y < lower_bound):
        posture = 'Jumping'
    elif (actual_mid_y > upper_bound):
        posture = 'Crouching'
    else:
        posture = 'Standing'
    if draw:
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)
    return output_image, posture


camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)
cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)
game_started = False
x_pos_index = 1
y_pos_index = 1
MID_Y = None
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame, results = detectPose(frame, pose_video, draw=game_started)
    if results.pose_landmarks:
        if game_started:
            frame, horizontal_position = checkLeftRight(frame, results, draw=True)
            if (horizontal_position == 'Left' and x_pos_index != 0) or (
                    horizontal_position == 'Center' and x_pos_index == 2):
                pyautogui.press('left')
                x_pos_index -= 1
            elif (horizontal_position == 'Right' and x_pos_index != 2) or (
                    horizontal_position == 'Center' and x_pos_index == 0):
                pyautogui.press('right')
                x_pos_index += 1

        else:
            game_started = True
            left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
            right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
            MID_Y = abs(right_y + left_y) // 2

        # Check if the intial y-coordinate of the mid-point of both shoulders of the person has a value.
        if MID_Y:
            frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
            if posture == 'Jumping' and y_pos_index == 1:
                pyautogui.press('up')
                y_pos_index += 1
            elif posture == 'Crouching' and y_pos_index == 1:
                pyautogui.press('down')
                y_pos_index -= 1
            elif posture == 'Standing' and y_pos_index != 1:
                y_pos_index = 1

    # Otherwise if the pose landmarks in the frame are not detected.
    else:
        pass
    # Display the frame.
    cv2.imshow('Subway Surfers with Pose Detection', frame)

    # Wait for 1ms. If a key is pressed, retrieve the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if (k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()