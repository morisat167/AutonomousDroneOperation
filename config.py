# config.py

# Models
SEG_MODEL_PATH = "yolo11m-seg.pt"   # obstacles segmentation
DET_MODEL_PATH = "yolo11m.pt"        # person + bicycle detection/tracking

# Detection / segmentation
DET_CONF = 0.30
SEG_CONF = 0.12
IMGSZ_DET = 640
IMGSZ_SEG = 640

# Goal (bicycle)
GOAL_CLASS_ID = 1
GOAL_LABEL = "bicycle"
GOAL_CONF_THRESH = 0.10
GOAL_REACH_DIST_PX = 35
GOAL_LOST_SEC = 1.0

# Flight behavior
FLY = True
GO_UP_CM = 180
SEARCH_FOR_GOAL = True

# RRT
RRT_MAX_ITER = 700
RRT_STEP = 18
RRT_GOAL_TOL = 22
GOAL_BIAS = 0.07
PATH_DEVIATION_TOL = 50

# Ground ROI
GROUND_MODE = "fixed"       # "fixed" or "dynamic"
GROUND_TOP_Y_RATIO = 0.25
GROUND_TOP_WIDTH_RATIO = 0.4
GROUND_DILATE = 0
GROUND_ERODE = 0

# Recording
SAVE_VIDEO = True
VIDEO_PATH = "mission_output.avi"
VIDEO_FPS = 20.0

# Drone control
MAX_LR = 40
MAX_FB = 35

KP_FB = 220.0
KD_FB = 80.0
DESIRED_AREA_RATIO = 0.10
AREA_TOL = 0.03

KP_LR = 0.12
KD_LR = 0.05
DEADBAND_X = 18

# User lock / red shirt acquisition
STICKY_USER_FRAMES = 25
STICKY_IOU_MIN = 0.15

RED_RATIO_THRESH = 0.20
RED_S_MIN = 80
RED_V_MIN = 60
