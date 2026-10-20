import gradio as gr
import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

robot = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

cube = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.55, 0, 0.025],
    globalScaling=1.0
)

p.changeVisualShape(cube, -1, rgbaColor=[1, 0.1, 0.1, 1])

# Panda 默认关节姿态
initial_joints = [0, -0.6, 0, -2.2, 0, 1.6, 0.8]
for i, angle in enumerate(initial_joints):
    p.resetJointState(robot, i, angle)

for _ in range(100):
    p.stepSimulation()

def render():
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.4, 0, 0.3],
        distance=1.5,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60, aspect=4/3, nearVal=0.1, farVal=5
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
    return Image.fromarray(rgb)

with gr.Blocks(title="Franka Panda Robot Simulator") as demo:
    gr.Markdown("# Franka Panda Robot Simulator")
    gr.Markdown("PyBullet simulation running on JHU BRTX")
    image = gr.Image(value=render, label="Robot Simulation", interactive=False)
    refresh = gr.Button("Refresh View")
    refresh.click(fn=render, outputs=image)

demo.launch(server_name="127.0.0.1", server_port=7862)
