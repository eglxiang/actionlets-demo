import threading
import gradio as gr
import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image

lock = threading.Lock()
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)

p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
cube = p.loadURDF("cube_small.urdf", [0.55, 0, 0.025])
p.changeVisualShape(cube, -1, rgbaColor=[1, 0.1, 0.1, 1])

HOME = [0, -0.6, 0, -2.2, 0, 1.6, 0.8]
EE_LINK = 11
history = []

def render():
    view = p.computeViewMatrixFromYawPitchRoll(
        [0.4, 0, 0.3], 1.5, 45, -30, 0, 2
    )
    proj = p.computeProjectionMatrixFOV(60, 4/3, 0.1, 5)
    _, _, rgba, _, _ = p.getCameraImage(
        640, 480, view, proj, renderer=p.ER_TINY_RENDERER
    )
    return Image.fromarray(
        np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
    )

def step():
    for _ in range(120):
        p.stepSimulation()

def get_xyz():
    return tuple(round(v, 3) for v in p.getLinkState(robot, EE_LINK)[4])

def reset():
    with lock:
        for i, angle in enumerate(HOME):
            p.resetJointState(robot, i, angle)
        for i in (9, 10):
            p.resetJointState(robot, i, 0.04)
        p.resetBasePositionAndOrientation(cube, [0.55, 0, 0.025], [0, 0, 0, 1])
        p.resetBaseVelocity(cube, [0, 0, 0], [0, 0, 0])
        history.clear()
        step()
        xyz = get_xyz()
        return render(), *xyz, "Reset complete", ""

def move(x, y, z):
    with lock:
        target = [float(x), float(y), float(z)]
        orientation = p.getLinkState(robot, EE_LINK)[5]
        joints = p.calculateInverseKinematics(
            robot, EE_LINK, target, orientation, maxNumIterations=200
        )
        for i in range(7):
            p.setJointMotorControl2(
                robot, i, p.POSITION_CONTROL,
                targetPosition=joints[i], force=200
            )
        for _ in range(480):
            p.stepSimulation()

        actual = get_xyz()
        entry = f"Move target={target}, actual={actual}"
        history.append(entry)
        return render(), entry, "\n".join(history)

def gripper(open_gripper):
    with lock:
        width = 0.04 if open_gripper else 0.0
        for i in (9, 10):
            p.setJointMotorControl2(
                robot, i, p.POSITION_CONTROL,
                targetPosition=width, force=50
            )
        step()
        contacts = p.getContactPoints(bodyA=robot, bodyB=cube)
        finger_contacts = [c for c in contacts if c[3] in (9, 10)]
        for link in (9, 10):
            points = p.getClosestPoints(
                bodyA=robot, bodyB=cube,
                distance=0.5, linkIndexA=link
            )
            distance = min((pt[8] for pt in points), default=None)
            print(f"[Finger Distance] link={link}, distance={distance}", flush=True)
        cube_pos = p.getBasePositionAndOrientation(cube)[0]
        print(
            f"[Grasp Debug] contacts={len(contacts)}, "
            f"finger_contacts={len(finger_contacts)}, "
            f"cube_pos={tuple(round(v, 3) for v in cube_pos)}",
            flush=True
        )
        entry = "Open gripper" if open_gripper else "Close gripper"
        history.append(entry)
        return render(), entry, "\n".join(history)


def pick_place_actionlet(target_x, target_y):
    """Execute a reusable pick-and-place sequence."""
    source = p.getBasePositionAndOrientation(cube)[0]
    sx, sy = source[:2]
    tx, ty = float(target_x), float(target_y)

    actions = [
        ("Open", lambda: gripper(True)),
        ("Approach", lambda: move(sx, sy, 0.05)),
        ("Grasp", lambda: gripper(False)),
        ("Lift", lambda: move(sx, sy, 0.25)),
        ("Transfer", lambda: move(tx, ty, 0.25)),
        ("Lower", lambda: move(tx, ty, 0.05)),
        ("Release", lambda: gripper(True)),
        ("Retreat", lambda: move(tx, ty, 0.25)),
    ]

    trace = []
    for name, action in actions:
        action()
        with lock:
            ee = p.getLinkState(robot, EE_LINK)[4]
            obj = p.getBasePositionAndOrientation(cube)[0]
            contacts = p.getContactPoints(bodyA=robot, bodyB=cube)
            record = {
                "stage": name,
                "ee_xyz": [round(v, 3) for v in ee],
                "object_xyz": [round(v, 3) for v in obj],
                "contacts": len(contacts),
            }
        trace.append(record)
        print(f"[Actionlet Trace] {record}", flush=True)
        yield render(), f"Executing: {name}", "\\n".join(history)

    with lock:
        final = p.getBasePositionAndOrientation(cube)[0]
        error = ((final[0] - tx)**2 + (final[1] - ty)**2)**0.5
        success = error < 0.05 and final[2] < 0.06
        result = (
            f"Pick-and-Place: {'SUCCESS' if success else 'FAILED'}; "
            f"final={tuple(round(v, 3) for v in final)}; "
            f"target=({tx:.3f}, {ty:.3f}); error={error:.3f}m"
        )
        history.append(result)
        yield render(), result, "\\n".join(history)


# Step-by-step Actionlet controller
step_state = {"actions": [], "index": 0}

def start_actionlet(target_x, target_y):
    with lock:
        source = p.getBasePositionAndOrientation(cube)[0]
    sx, sy = source[:2]
    tx, ty = float(target_x), float(target_y)

    step_state["actions"] = [
        ("Open", lambda: gripper(True)),
        ("Approach", lambda: move(sx, sy, 0.05)),
        ("Grasp", lambda: gripper(False)),
        ("Lift", lambda: move(sx, sy, 0.25)),
        ("Transfer", lambda: move(tx, ty, 0.25)),
        ("Lower", lambda: move(tx, ty, 0.05)),
        ("Release", lambda: gripper(True)),
        ("Retreat", lambda: move(tx, ty, 0.25)),
    ]
    step_state["index"] = 0

    with lock:
        return render(), "Ready: click Next Step", "\\n".join(history)

def next_actionlet_step():
    actions = step_state["actions"]
    index = step_state["index"]

    if not actions:
        with lock:
            return render(), "Click Start Actionlet first", "\\n".join(history)

    if index >= len(actions):
        with lock:
            return render(), "Actionlet already completed", "\\n".join(history)

    name, action = actions[index]
    action()
    step_state["index"] += 1

    with lock:
        obj = p.getBasePositionAndOrientation(cube)[0]
        contacts = p.getContactPoints(bodyA=robot, bodyB=cube)
        status = (
            f"Step {index + 1}/{len(actions)}: {name}; "
            f"cube={tuple(round(v, 3) for v in obj)}; "
            f"contacts={len(contacts)}"
        )
        print(f"[Step Actionlet] {status}", flush=True)
        return render(), status, "\\n".join(history)

with gr.Blocks(title="Franka Panda Robot Simulator") as demo:
    gr.Markdown("# Franka Panda Robot Simulator")
    gr.Markdown("PyBullet | Inverse Kinematics | Action History")

    image = gr.Image(label="Robot Simulation", interactive=False)

    with gr.Row():
        x = gr.Number(label="X (meters)")
        y = gr.Number(label="Y (meters)")
        z = gr.Number(label="Z (meters)")

    with gr.Row():
        move_btn = gr.Button("Move to XYZ", variant="primary")
        open_btn = gr.Button("Open Gripper")
        close_btn = gr.Button("Close Gripper")
        reset_btn = gr.Button("Reset")

    gr.Markdown("### Reusable Pick-and-Place Actionlet")
    with gr.Row():
        target_x = gr.Number(label="Place X (meters)", value=0.45)
        target_y = gr.Number(label="Place Y (meters)", value=0.20)
    run_actionlet_btn = gr.Button("Run Pick-and-Place", variant="primary")

    gr.Markdown("### Step-by-Step Actionlet")
    with gr.Row():
        start_step_btn = gr.Button("Start Actionlet")
        next_step_btn = gr.Button("Next Step")

    status = gr.Textbox(label="Status")
    log = gr.Textbox(label="Action History", lines=6)

    run_actionlet_btn.click(
        pick_place_actionlet,
        inputs=[target_x, target_y],
        outputs=[image, status, log]
    )
    start_step_btn.click(
        start_actionlet,
        inputs=[target_x, target_y],
        outputs=[image, status, log]
    )
    next_step_btn.click(
        next_actionlet_step,
        outputs=[image, status, log]
    )
    move_btn.click(move, [x, y, z], [image, status, log])
    open_btn.click(lambda: gripper(True), outputs=[image, status, log])
    close_btn.click(lambda: gripper(False), outputs=[image, status, log])
    reset_btn.click(reset, outputs=[image, x, y, z, status, log])
    demo.load(reset, outputs=[image, x, y, z, status, log])

demo.launch(server_name="127.0.0.1", server_port=7862)
