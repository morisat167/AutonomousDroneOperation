# AutonomousDroneOperation
This project implements a real-time autonomous navigation system for a DJI Tello drone using computer vision, path planning, and closed-loop control. The system detects a human user and a bicycle target using a YOLO-based perception pipeline, constructs an obstacle map via instance segmentation, and plans a collision-free route using a Rapidly-Exploring Random Tree (RRT) algorithm.

The drone follows the user using a vision-based PD controller while continuously replanning a safe path to the target in dynamic outdoor environments. All perception, planning, and control algorithms run in real time on a laptop computer, while the drone provides live video streaming and executes motion commands via the DJI SDK interface.

