"""Franka block stacking driven by an LLM (via LLM Gateway) or by a trained policy, recorded to ROS 2 MCAP bags.

Each episode spawns red, green and blue blocks at random positions and asks for one of the six
"stack the <A> block on the <B> block" tasks.

``--mode llm``: every step the LLM receives the task, the measured robot/block state, the step history and
both camera images, and answers with one motion primitive (move_to / open_gripper / close_gripper / done)
as structured JSON. RMPflow plus the joint gripper controller execute it.

``--mode llm --observation vision``: the LLM gets no block poses, only the two camera images (with pixel rulers) and
the robot's proprioception. It can call ``locate(camera, u, v)``, which deprojects the depth image at that pixel into
a world point, like an RGB-D robot with calibrated cameras. The images it was shown are saved to
``<episode>/llm_images/``; each locate step also stores its ground-truth error under ``debug`` (never shown to it).

``--mode policy``: a policy server (``train_policy.py serve``) receives camera images and joint positions
at the camera rate and returns joint position targets.

Every episode is recorded by ``ros2 bag record`` into ``<out>/<episode>/`` (MCAP, zstd) together with
``episode.json`` (task, seed, per-step prompts/responses/results, success). In addition to the topics of
``franka_pick_place_ros2.py`` (minus point clouds, which are derivable from depth + camera_info) the bags
contain:
    /task                 std_msgs/String (transient local)
    /llm/step             std_msgs/String (JSON request summary + response + execution result per step)
    /blocks/<color>/pose  geometry_msgs/PoseStamped
    /episode/result       std_msgs/String (JSON, transient local, published at the end)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default=None, help="Directory that receives one sub-directory per episode")
parser.add_argument("--jobs", default=None,
                    help="Serve episode requests from <jobs>/queue instead of running a batch (Franka LLM Lab)")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0, help="Episode i uses seed + i")
parser.add_argument("--mode", choices=("llm", "policy"), default="llm")
parser.add_argument("--model", default="gpt-6-astra")
parser.add_argument("--gateway", default="https://llm-gateway.simulprod.com/llm-gateway-internal/v1")
parser.add_argument("--token-file", default=os.path.expanduser("~/.config/llm-gateway/lca_token"))
parser.add_argument("--max-steps", type=int, default=None,
                    help="LLM action budget per episode (default 24, or 40 with --observation vision)")
parser.add_argument("--observation", choices=("state", "vision"), default="state",
                    help="state: the prompt lists measured block poses. vision: the LLM only sees the camera images "
                         "and the robot's own proprioception, and can query the depth cameras with locate(u, v)")
parser.add_argument("--policy-address", default="127.0.0.1:47710")
parser.add_argument("--policy-seconds", type=float, default=30.0, help="Sim time budget per policy episode")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--camera-hz", type=float, default=15.0)
parser.add_argument("--no-record", action="store_true", help="Skip the ROS 2 bag recorder")
parser.add_argument("--table", choices=("grid", "wood", "plain"), default="grid",
                    help="Table appearance: the default grid ground plane, a wood texture, or a solid color")
args, _ = parser.parse_known_args()
if args.max_steps is None:
    args.max_steps = 40 if args.observation == "vision" else 24

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": args.gpu,
                                "physics_gpu": args.gpu})

import isaacsim.core.experimental.utils.app as app_utils

app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.robot_motion.examples")
simulation_app.update()

from enum import Enum
from multiprocessing.connection import Client

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.graph.core as og
import omni.usd
import rclpy
import usdrt.Sdf
import warp as wp
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.utils.semantics import add_labels
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import GripperCommand, JointGripperController, ManipulationScenario
from isaacsim.sensors.experimental.rtx import RtxCamera
from PIL import Image as PILImage
from pxr import Gf, UsdGeom
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

PHYSICS_DT = 1.0 / 60.0
WORLD_FRAME = "World"
ROBOT_PATH = "/World/robot"
HAND_PATH = f"{ROBOT_PATH}/panda_hand"
SCENE_CAMERA_PATH = "/World/scene_camera"
WRIST_CAMERA_PATH = f"{HAND_PATH}/wrist_camera"
BLOCK_SIZE = 0.05
COLORS = {"red": (0.8, 0.1, 0.1), "green": (0.1, 0.7, 0.1), "blue": (0.1, 0.25, 0.85)}
TASKS = [(a, b) for a in COLORS for b in COLORS if a != b]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
WORKSPACE = {"x": (0.30, 0.62), "y": (-0.28, 0.28)}
MOVE_LIMITS = {"x": (0.15, 0.8), "y": (-0.5, 0.5), "z": (0.012, 0.5)}
MOVE_TIMEOUT = 4.0
GRIPPER_TIMEOUT = 1.5
POSITION_TOLERANCE = 0.008
SETTLE_SECONDS = 1.0
LIVE_FRAME_INTERVAL = 6  # physics steps between live preview frames in --jobs mode

SYSTEM_PROMPT = f"""You control a simulated Franka Panda arm with a parallel-jaw gripper above a table.
World frame: meters, +z up, the table surface is z=0, the robot base is at the origin.
Blocks are {BLOCK_SIZE:.2f} m cubes. A block resting on the table has its center at z={BLOCK_SIZE / 2:.3f}; a block resting
on top of another block has its center at z={1.5 * BLOCK_SIZE:.3f}.
The gripper always points straight down. move_to(x, y, z) moves the grasp point (midway between the fingertips)
to that position with a collision-unaware motion controller, so the hand will knock over any block it passes
through: move above a block before descending onto it, and lift before traveling.
The gripper fingers open to 8 cm. To grasp a block, open the gripper, put the grasp point at the block center,
then close the gripper. Closing on a block leaves the fingers about 2.5 cm each from closed; closing on nothing
leaves them at 0. To place a block, move it to where its center should be (or slightly above) and open.
Each turn, choose exactly one action:
  move_to: x, y, z are the target grasp point.
  open_gripper / close_gripper: x, y, z are ignored (send 0).
  done: the task is complete (x, y, z ignored).
Look at the measured state and the result of your previous action before choosing the next one."""

SCENE_CAMERA_EYE = (1.35, 0.0, 0.85)
SCENE_CAMERA_TARGET = (0.45, 0.0, 0.05)

VISION_SYSTEM_PROMPT = f"""You control a Franka Panda arm with a parallel-jaw gripper above a table, like a real robot:
you do NOT get object positions. You only see two RGB-D cameras and the robot's own proprioception.
World frame: meters, +z up, the table surface is z=0, the robot base is at the origin, +x points away from the robot
base, +y to the robot's left.
Cameras (images are {{width}}x{{height}} pixels, u to the right, v down, origin at the top-left corner):
  scene camera: fixed, at ({SCENE_CAMERA_EYE[0]}, {SCENE_CAMERA_EYE[1]}, {SCENE_CAMERA_EYE[2]}) looking at \
({SCENE_CAMERA_TARGET[0]}, {SCENE_CAMERA_TARGET[1]}, {SCENE_CAMERA_TARGET[2]}), i.e. facing the robot from the far \
side of the table. In it, +y is to the right of the image and +x is toward the bottom of the image.
  wrist camera: mounted on the hand, looking straight down past the fingers (moves with the gripper).
Pixel rulers are drawn along the top and left image edges to help you read pixel coordinates.
Blocks are {BLOCK_SIZE:.2f} m cubes. A block resting on the table has its center at z={BLOCK_SIZE / 2:.3f} and its top \
face at z={BLOCK_SIZE:.3f}; a block resting on top of another block has its center at z={1.5 * BLOCK_SIZE:.3f}.
The gripper always points straight down. move_to(x, y, z) moves the grasp point (midway between the fingertips)
to that position with a collision-unaware motion controller, so the hand will knock over any block it passes
through: move above a block before descending onto it, and lift before traveling.
The gripper fingers open to 8 cm. To grasp a block, open the gripper, put the grasp point at the block center,
then close the gripper. Closing on a block leaves the fingers about 2.5 cm each from closed; closing on nothing
leaves them at 0. To place a block, move it to where its center should be (or slightly above) and open.
To find where things are, use locate: it reads the depth camera at pixel (u, v) of the scene or wrist image you were
just shown and returns the 3D world point of the first surface seen through that pixel (for the top face of a block
on the table, z is about {BLOCK_SIZE:.3f}; for the table, z is about 0). It does not move the robot and takes no time.
Point at the middle of a block's top face for the best estimate of its center x, y. The wrist camera gives more
precise readings when the hand is above the object.
Each turn, choose exactly one action:
  locate: camera is "scene" or "wrist", u and v are the pixel (x, y, z ignored).
  move_to: x, y, z are the target grasp point (camera, u, v ignored).
  open_gripper / close_gripper: all other fields ignored (send 0 / "scene").
  done: the task is complete.
Check the images and the result of your previous action before choosing the next one."""

VISION_ACTION_SCHEMA = {
    "name": "robot_action",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reasoning", "action", "x", "y", "z", "camera", "u", "v"],
        "properties": {
            "reasoning": {"type": "string",
                          "description": "One or two sentences on what you see and why this action is next."},
            "action": {"type": "string", "enum": ["locate", "move_to", "open_gripper", "close_gripper", "done"]},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "camera": {"type": "string", "enum": ["scene", "wrist"]},
            "u": {"type": "number"},
            "v": {"type": "number"},
        },
    },
}

ACTION_SCHEMA = {
    "name": "robot_action",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reasoning", "action", "x", "y", "z"],
        "properties": {
            "reasoning": {"type": "string", "description": "One or two sentences on why this action is next."},
            "action": {"type": "string", "enum": ["move_to", "open_gripper", "close_gripper", "done"]},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
        },
    },
}


def sim_stamp(seconds: float) -> Time:
    sec = int(math.floor(seconds))
    return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)) % 1_000_000_000)


def block_path(color: str) -> str:
    return f"/World/Block_{color}"


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[0], q[1:]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + w * v)


def define_camera(path: str, focal_length: float = 18.0) -> UsdGeom.Camera:
    camera = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), path)
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetVerticalApertureAttr().Set(20.955 * args.height / args.width)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return camera


def build_scene() -> tuple[ManipulationScenario, dict[str, RigidPrim]]:
    scenario = ManipulationScenario("franka", robot_prim_path=ROBOT_PATH)
    scenario.setup_scene()
    blocks = {}
    for i, (color, rgb) in enumerate(COLORS.items()):
        cube = Cube(block_path(color), positions=[(0.4, -0.2 + 0.2 * i, BLOCK_SIZE / 2)], sizes=BLOCK_SIZE,
                    colors=[list(rgb)])
        blocks[color] = RigidPrim(cube.paths)
        GeomPrim(cube.paths, apply_collision_apis=True)
        add_labels(block_path(color), labels=f"{color}_block")
    add_labels(ROBOT_PATH, labels="franka")
    add_labels("/World/ground_plane", labels="ground")

    scene_camera = define_camera(SCENE_CAMERA_PATH)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*SCENE_CAMERA_EYE), Gf.Vec3d(*SCENE_CAMERA_TARGET), Gf.Vec3d(0, 0, 1))
    scene_camera.AddTransformOp().Set(view.GetInverse())

    # USD cameras look down -Z; flip so the wrist camera looks along the hand's +Z toward the fingers
    wrist_camera = define_camera(WRIST_CAMERA_PATH, focal_length=12.0)
    xform = UsdGeom.XformCommonAPI(wrist_camera)
    xform.SetTranslate(Gf.Vec3d(0.06, 0.0, 0.02))
    xform.SetRotate(Gf.Vec3f(180.0, 0.0, 90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    # RtxCamera rewrites the xform ops, so the poses above must be authored first
    for path in (SCENE_CAMERA_PATH, WRIST_CAMERA_PATH):
        RtxCamera(path, tick_rate=args.camera_hz)
    build_table()
    return scenario, blocks


TABLE_PATH = "/World/table"
TABLE_TEXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "table_wood.jpg")
TABLE_TILE_M = 1.2  # one repeat of the wood texture covers 1.2 m (planks about 14 cm wide)
TABLE_PLAIN_COLOR = (0.52, 0.47, 0.42)


def build_table() -> None:
    """Visual-only table top at z=0 over the ground plane (whose Plane prim keeps providing the collision)."""
    from pxr import Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()
    x0, x1, y0, y1 = -1.6, 2.0, -1.8, 1.8  # covers everything the scene camera can see
    mesh = UsdGeom.Mesh.Define(stage, TABLE_PATH)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    mesh.GetPointsAttr().Set([Gf.Vec3f(x, y, 0.0) for x, y in corners])
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(y / TABLE_TILE_M, x / TABLE_TILE_M) for x, y in corners])  # planks run along x
    add_labels(TABLE_PATH, labels="table")

    def material(name: str, texture: str | None, color=(1.0, 1.0, 1.0)) -> UsdShade.Material:
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        surface = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if texture:
            reader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/StReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            image = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Texture")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
            image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result")
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                image.ConnectableAPI(), "rgb")
        else:
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        return mat

    TABLE_MATERIALS["wood"] = material("TableWood", TABLE_TEXTURE)
    TABLE_MATERIALS["plain"] = material("TablePlain", None, TABLE_PLAIN_COLOR)
    set_table(args.table)


TABLE_MATERIALS: dict = {}


def set_table(kind: str) -> None:
    """Switch the table appearance between episodes: grid (default ground plane), wood or plain."""
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    table = stage.GetPrimAtPath(TABLE_PATH)
    ground_meshes = [p for p in stage.GetPrimAtPath("/World/ground_plane").GetChildren() if p.IsA(UsdGeom.Mesh)]
    imageable = UsdGeom.Imageable(table)
    if kind == "grid":
        imageable.MakeInvisible()
        for prim in ground_meshes:
            UsdGeom.Imageable(prim).MakeVisible()
        return
    UsdShade.MaterialBindingAPI.Apply(table).Bind(TABLE_MATERIALS[kind])
    imageable.MakeVisible()
    for prim in ground_meshes:
        UsdGeom.Imageable(prim).MakeInvisible()


def build_state_graph() -> None:
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_State", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
                ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
                ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
                ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
                ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
                ("ReadJointState.outputs:stageMetersPerUnit", "PublishJointState.inputs:stageMetersPerUnit"),
                ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
            ],
            keys.SET_VALUES: [
                ("ReadJointState.inputs:prim", [usdrt.Sdf.Path(ROBOT_PATH)]),
                ("PublishTF.inputs:parentPrim", [usdrt.Sdf.Path("/World")]),
                ("PublishTF.inputs:targetPrims",
                 [usdrt.Sdf.Path(p) for p in (ROBOT_PATH, SCENE_CAMERA_PATH, WRIST_CAMERA_PATH,
                                              *(block_path(c) for c in COLORS))]),
            ],
        },
    )


def build_camera_graph(name: str, camera_path: str) -> None:
    keys = og.Controller.Keys
    frame_id = f"{name}_optical"
    helpers = [
        ("rgb", "rgb"),
        ("depth", "depth"),
        ("instance_segmentation", "instance_segmentation"),
        ("semantic_segmentation", "semantic_segmentation"),
        ("bbox_2d", "bbox_2d_tight"),
        ("bbox_3d", "bbox_3d"),
    ]
    nodes = [("OnTick", "omni.graph.action.OnTick"),
             ("Context", "isaacsim.ros2.bridge.ROS2Context"),
             ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
             ("CameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
    connect = [("OnTick.outputs:tick", "RenderProduct.inputs:execIn"),
               ("RenderProduct.outputs:execOut", "CameraInfo.inputs:execIn"),
               ("RenderProduct.outputs:renderProductPath", "CameraInfo.inputs:renderProductPath"),
               ("Context.outputs:context", "CameraInfo.inputs:context")]
    values = [("RenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
              ("RenderProduct.inputs:width", args.width),
              ("RenderProduct.inputs:height", args.height),
              ("CameraInfo.inputs:nodeNamespace", name),
              ("CameraInfo.inputs:frameId", frame_id),
              ("CameraInfo.inputs:topicName", "camera_info")]
    for topic, helper_type in helpers:
        node = f"Helper_{topic}"
        nodes.append((node, "isaacsim.ros2.bridge.ROS2CameraHelper"))
        connect += [("RenderProduct.outputs:execOut", f"{node}.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", f"{node}.inputs:renderProductPath"),
                    ("Context.outputs:context", f"{node}.inputs:context")]
        values += [(f"{node}.inputs:nodeNamespace", name),
                   (f"{node}.inputs:frameId", frame_id),
                   (f"{node}.inputs:topicName", topic),
                   (f"{node}.inputs:type", helper_type)]
        if helper_type in ("instance_segmentation", "semantic_segmentation"):
            values += [(f"{node}.inputs:enableSemanticLabels", True),
                       (f"{node}.inputs:semanticLabelsTopicName", helper_type.replace("segmentation", "labels"))]
    graph, _, _, _ = og.Controller.edit(
        {"graph_path": f"/ROS2_{name}", "evaluator_name": "push",
         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connect, keys.SET_VALUES: values},
    )
    og.Controller.evaluate_sync(graph)


class RosIO:
    """Publishes episode metadata, commands and block poses, and keeps the latest camera images."""

    def __init__(self, scenario: ManipulationScenario, blocks: dict[str, RigidPrim], urdf: str):
        self.scenario = scenario
        self.blocks = blocks
        self.node = rclpy.create_node("franka_llm_episode")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_description = self.node.create_publisher(String, "robot_description", latched)
        self.pub_tf_static = self.node.create_publisher(TFMessage, "tf_static", latched)
        self.pub_task = self.node.create_publisher(String, "task", latched)
        self.pub_result = self.node.create_publisher(String, "episode/result", latched)
        self.pub_step = self.node.create_publisher(String, "llm/step", 10)
        self.pub_commands = self.node.create_publisher(JointState, "joint_commands", 10)
        self.pub_ee = self.node.create_publisher(PoseStamped, "ee_pose", 10)
        self.pub_blocks = {c: self.node.create_publisher(PoseStamped, f"blocks/{c}/pose", 10) for c in blocks}
        self.images: dict[str, Image] = {}
        self.depths: dict[str, Image] = {}
        for camera in ("scene_camera", "wrist_camera"):
            self.node.create_subscription(Image, f"{camera}/rgb",
                                          lambda msg, camera=camera: self.images.__setitem__(camera, msg), 2)
            self.node.create_subscription(Image, f"{camera}/depth",
                                          lambda msg, camera=camera: self.depths.__setitem__(camera, msg), 2)
        self.urdf = urdf
        self.publish_latched()

    def publish_latched(self) -> None:
        self.pub_description.publish(String(data=self.urdf))
        transforms = []
        for camera in ("scene_camera", "wrist_camera"):
            t = TransformStamped()
            t.header.frame_id = camera
            t.child_frame_id = f"{camera}_optical"
            t.transform.rotation.x = 1.0
            t.transform.rotation.w = 0.0
            transforms.append(t)
        self.pub_tf_static.publish(TFMessage(transforms=transforms))

    @staticmethod
    def _pose(stamp: Time, position, orientation_wxyz) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = WORLD_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in position)
        w, x, y, z = (float(v) for v in orientation_wxyz)
        msg.pose.orientation.w, msg.pose.orientation.x = w, x
        msg.pose.orientation.y, msg.pose.orientation.z = y, z
        return msg

    def publish(self, sim_time: float, command: np.ndarray) -> None:
        stamp = sim_stamp(sim_time)
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in command]
        self.pub_commands.publish(msg)
        ee_positions, ee_orientations = self.scenario.read_tool_pose()
        self.pub_ee.publish(self._pose(stamp, ee_positions[0], ee_orientations[0]))
        for color, prim in self.blocks.items():
            positions, orientations = prim.get_world_poses()
            self.pub_blocks[color].publish(self._pose(stamp, positions.numpy()[0], orientations.numpy()[0]))
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def latest_rgb(self, camera: str, wait: float = 1.0) -> np.ndarray | None:
        deadline = time.time() + wait
        while camera not in self.images and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        msg = self.images.get(camera)
        if msg is None:
            return None
        return np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[..., :3]

    def drain(self) -> None:
        for _ in range(40):
            rclpy.spin_once(self.node, timeout_sec=0.005)

    def latest_depth(self, camera: str) -> np.ndarray | None:
        msg = self.depths.get(camera)
        if msg is None:
            return None
        return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)

    def close(self) -> None:
        self.node.destroy_node()


class Recorder:
    """Runs ``ros2 bag record`` (system Jazzy, python3.12) for one episode in a clean environment."""

    def __init__(self, output: str):
        env = {k: os.environ[k] for k in ("HOME", "USER", "ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE",
                                          "RMW_IMPLEMENTATION") if k in os.environ}
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        command = ("source /opt/ros/jazzy/setup.bash && exec python3.12 /opt/ros/jazzy/bin/ros2 bag record --all "
                   "--exclude-regex '.*/points$' --storage mcap --storage-preset-profile zstd_fast "
                   "--max-cache-size 536870912 --use-sim-time --output \"$0\"")
        self.process = subprocess.Popen(["bash", "-c", command, output], env=env, start_new_session=True,
                                        stdout=open(output + ".recorder.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(3.0)

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)


class Gripper(Enum):
    OPEN = 0
    CLOSE = 1


class Robot:
    """Executes motion primitives with RMPflow and the joint gripper controllers."""

    def __init__(self, scenario: ManipulationScenario, io: RosIO):
        self.scenario = scenario
        self.io = io
        config = scenario.robot_config
        self.tool_frame = config.tool.controller_frame
        self.grasp_orientation = np.asarray(config.grasp_orientation, dtype=np.float64)
        self.grasp_offset = quat_rotate(self.grasp_orientation, -np.asarray(config.tool.controller_to_grasp_position))
        gripper = config.gripper
        gripper_args = {
            "robot_joint_space": scenario.joint_space,
            "joint_names": gripper.joint_names,
            "open_positions": gripper.open_positions,
            "closed_positions": gripper.closed_positions,
            "position_tolerance": gripper.position_tolerance,
            "velocity_tolerance": gripper.velocity_tolerance,
            "minimum_close_fraction": gripper.minimum_close_fraction,
        }
        self.grippers = {
            Gripper.OPEN: JointGripperController(**gripper_args, command=GripperCommand.OPEN),
            Gripper.CLOSE: JointGripperController(**gripper_args, command=GripperCommand.CLOSE),
        }
        cumotion_robot = load_cumotion_supported_robot("franka")
        arm = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=scenario.world_interface,
            robot_joint_space=scenario.joint_space,
            robot_site_space=scenario.site_space,
            tool_frame=self.tool_frame,
        )
        self.controller = mg.SelectableController(
            {g: mg.CombinedController([arm, self.grippers[g]]) for g in Gripper}, Gripper.OPEN)
        self.joint_index = [scenario.joint_space.index(n) for n in JOINT_NAMES]
        self.t = 0.0
        self.live_dir: str | None = None
        self.frame_counter = 0
        self.gripper_state = Gripper.OPEN
        self.target = self.grasp_position()
        self.command = self.joint_positions()

    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.scenario.articulation.get_dof_positions().numpy()[0], dtype=np.float64)[
            [self.scenario.articulation.dof_names.index(n) for n in JOINT_NAMES]]

    def grasp_position(self) -> np.ndarray:
        position = np.asarray(self.scenario.read_tool_pose()[0][0], dtype=np.float64)
        return position - self.grasp_offset

    def setpoint(self) -> mg.RobotState:
        controller_position = self.target + self.grasp_offset
        return mg.RobotState(sites=mg.SpatialState.from_name(
            spatial_space=self.scenario.site_space,
            positions=([self.tool_frame], wp.array([controller_position], dtype=wp.float32, device="cpu")),
            orientations=([self.tool_frame], wp.array([self.grasp_orientation], dtype=wp.float32, device="cpu")),
        ))

    def reset(self) -> None:
        self.gripper_state = Gripper.OPEN
        self.controller.set_next_controller(Gripper.OPEN)
        self.target = self.grasp_position()
        if not self.controller.reset(self.scenario.read_robot_state(), self.setpoint(), self.t):
            raise RuntimeError("Controller reset failed")
        self.command = self.joint_positions()

    def step(self, desired: mg.RobotState | None = None) -> None:
        self.t += PHYSICS_DT
        if desired is not None and desired.joints is not None and desired.joints.positions is not None:
            names = list(desired.joints.position_names)
            values = desired.joints.positions.numpy().reshape(-1)
            for name, value in zip(names, values):
                if name in JOINT_NAMES:
                    self.command[JOINT_NAMES.index(name)] = float(value)
        simulation_app.update()
        self.io.publish(SimulationManager.get_simulation_time(), self.command)
        self.frame_counter += 1
        if self.live_dir and self.frame_counter % LIVE_FRAME_INTERVAL == 0:
            self.write_live_frames()

    def write_live_frames(self) -> None:
        for camera in ("scene_camera", "wrist_camera"):
            rgb = self.io.latest_rgb(camera, wait=0.0)
            if rgb is None:
                continue
            path = os.path.join(self.live_dir, camera.split("_")[0] + ".jpg")
            PILImage.fromarray(rgb).save(path + ".tmp.jpg", format="JPEG", quality=80)
            os.replace(path + ".tmp.jpg", path)
        write_json(os.path.join(self.live_dir, "state.json"), {
            "sim_time": round(SimulationManager.get_simulation_time(), 3),
            "grasp_point": [round(float(v), 4) for v in self.grasp_position()],
            "target": [round(float(v), 4) for v in self.target],
            "gripper": self.gripper_state.name.lower(),
            "fingers": [round(float(v), 4) for v in self.joint_positions()[7:]],
        })

    def control_step(self) -> mg.RobotState:
        self.scenario.sync_world()
        desired = self.controller.forward(self.scenario.read_robot_state(), self.setpoint(), self.t + PHYSICS_DT)
        if desired is None:
            raise RuntimeError("RMPflow returned no command")
        self.scenario.apply_robot_state(desired)
        self.step(desired)
        return desired

    def move_to(self, target: np.ndarray) -> dict:
        self.target = target
        start = self.t
        stable = 0
        while self.t - start < MOVE_TIMEOUT:
            self.control_step()
            error = float(np.linalg.norm(self.grasp_position() - target))
            stable = stable + 1 if error < POSITION_TOLERANCE else 0
            if stable >= 6:
                break
        error = float(np.linalg.norm(self.grasp_position() - target))
        return {"reached": error < 2 * POSITION_TOLERANCE, "position_error_m": round(error, 4),
                "duration_s": round(self.t - start, 2)}

    def actuate(self, state: Gripper) -> dict:
        self.gripper_state = state
        self.controller.set_next_controller(state)
        start = self.t
        while self.t - start < GRIPPER_TIMEOUT:
            self.control_step()
            if self.t - start > 0.3 and self.grippers[state].is_complete(self.scenario.read_robot_state()):
                break
        fingers = self.joint_positions()[7:]
        return {"finger_positions_m": [round(float(v), 4) for v in fingers], "duration_s": round(self.t - start, 2)}

    def hold(self, seconds: float) -> None:
        for _ in range(round(seconds / PHYSICS_DT)):
            self.control_step()


def block_poses(blocks: dict[str, RigidPrim]) -> dict[str, np.ndarray]:
    return {c: np.asarray(p.get_world_poses()[0].numpy()[0], dtype=np.float64) for c, p in blocks.items()}


def randomize_blocks(blocks: dict[str, RigidPrim], rng: np.random.Generator) -> None:
    placed: list[np.ndarray] = []
    for color, prim in blocks.items():
        for _ in range(1000):
            xy = np.array([rng.uniform(*WORKSPACE["x"]), rng.uniform(*WORKSPACE["y"])])
            if all(np.linalg.norm(xy - p) > 0.12 for p in placed):
                break
        placed.append(xy)
        yaw = rng.uniform(-0.3, 0.3)
        prim.set_world_poses(positions=[[xy[0], xy[1], BLOCK_SIZE / 2]],
                             orientations=[[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
        prim.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])


def evaluate(blocks: dict[str, RigidPrim], task: tuple[str, str], initial: dict[str, np.ndarray]) -> dict:
    poses = block_poses(blocks)
    top, bottom = poses[task[0]], poses[task[1]]
    xy_error = float(np.linalg.norm(top[:2] - bottom[:2]))
    stacked = xy_error < 0.6 * BLOCK_SIZE and abs(top[2] - bottom[2] - BLOCK_SIZE) < 0.01 \
        and abs(bottom[2] - BLOCK_SIZE / 2) < 0.01
    others_moved = {c: round(float(np.linalg.norm(poses[c][:2] - initial[c][:2])), 3)
                    for c in blocks if c != task[0]}
    return {"success": bool(stacked), "xy_error_m": round(xy_error, 4),
            "height_diff_m": round(float(top[2] - bottom[2]), 4),
            "block_positions": {c: [round(float(v), 4) for v in p] for c, p in poses.items()},
            "displacement_of_other_blocks_m": others_moved}


def encode_jpeg(rgb: np.ndarray | None, size: tuple[int, int] | None = (384, 288), ruler: bool = False) -> str | None:
    if rgb is None:
        return None
    image = PILImage.fromarray(rgb)
    if size is not None:
        image = image.resize(size)
    if ruler:
        image = draw_ruler(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def draw_ruler(image: PILImage.Image) -> PILImage.Image:
    """Pixel ticks along the top and left edges (every 20 px, labeled every 100 px)."""
    from PIL import ImageDraw

    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for u in range(0, width, 20):
        length = 12 if u % 100 == 0 else 5
        draw.line([(u, 0), (u, length)], fill=(255, 255, 0), width=1)
        if u % 100 == 0 and u > 0:
            draw.text((u + 2, 2), str(u), fill=(255, 255, 0))
    for v in range(0, height, 20):
        length = 12 if v % 100 == 0 else 5
        draw.line([(0, v), (length, v)], fill=(255, 255, 0), width=1)
        if v % 100 == 0 and v > 0:
            draw.text((2, v + 2), str(v), fill=(255, 255, 0))
    return image


class CameraModel:
    """Pinhole model of a USD camera (looks down -Z, +Y up) rendered at args.width x args.height."""

    def __init__(self, path: str, focal_length: float):
        from isaacsim.core.experimental.prims import XformPrim

        self.prim = XformPrim(path)
        self.fx = focal_length / 20.955 * args.width
        self.cx, self.cy = args.width / 2.0, args.height / 2.0

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self.prim.get_world_poses()
        return (np.asarray(positions.numpy()[0], dtype=np.float64),
                np.asarray(orientations.numpy()[0], dtype=np.float64))

    def deproject(self, u: float, v: float, depth: float, pose) -> np.ndarray:
        position, orientation = pose
        local = np.array([(u + 0.5 - self.cx) / self.fx * depth, -(v + 0.5 - self.cy) / self.fx * depth, -depth])
        return position + quat_rotate(orientation, local)

    def project(self, point: np.ndarray, pose) -> tuple[float, float, float]:
        position, orientation = pose
        inverse = np.array([orientation[0], *(-orientation[1:])])
        local = quat_rotate(inverse, np.asarray(point, dtype=np.float64) - position)
        depth = -local[2]
        return (local[0] / depth * self.fx + self.cx - 0.5, -local[1] / depth * self.fx + self.cy - 0.5, depth)


class Perception:
    """What a real RGB-D robot has: per-camera RGB + depth frames and calibrated camera poses.

    ``snapshot`` freezes the frames shown to the LLM so ``locate`` reads the same depth image the LLM looked at.
    """

    CAMERAS = {"scene": ("scene_camera", SCENE_CAMERA_PATH, 18.0), "wrist": ("wrist_camera", WRIST_CAMERA_PATH, 12.0)}

    def __init__(self, io: RosIO):
        self.io = io
        self.models = {k: CameraModel(path, focal) for k, (_, path, focal) in self.CAMERAS.items()}
        self.frames: dict[str, dict] = {}
        self.image_dir: str | None = None

    def snapshot(self, robot: Robot) -> None:
        robot.hold(0.25)  # stationary for a few camera frames so RGB, depth and camera pose agree
        self.io.drain()
        for key, (topic, _, _) in self.CAMERAS.items():
            self.frames[key] = {"rgb": self.io.latest_rgb(topic), "depth": self.io.latest_depth(topic),
                                "pose": self.models[key].pose()}

    def images(self, step: int) -> list[str]:
        encoded = [encode_jpeg(self.frames[k]["rgb"], size=None, ruler=True) for k in ("scene", "wrist")]
        if self.image_dir:
            os.makedirs(self.image_dir, exist_ok=True)
            for key, data in zip(("scene", "wrist"), encoded):
                if data:
                    with open(os.path.join(self.image_dir, f"step{step:02d}_{key}.jpg"), "wb") as f:
                        f.write(base64.b64decode(data))
        return encoded

    def locate(self, camera: str, u: float, v: float) -> dict:
        frame = self.frames.get(camera)
        if frame is None or frame["depth"] is None:
            return {"error": f"no {camera} depth image"}
        depth_image = frame["depth"]
        height, width = depth_image.shape
        if not (0 <= u < width and 0 <= v < height):
            return {"error": f"pixel ({u:.0f}, {v:.0f}) is outside the {width}x{height} image"}
        # median over a 5x5 window ignores single invalid pixels at edges
        ui, vi = int(u), int(v)
        window = depth_image[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
        window = window[np.isfinite(window) & (window > 0)]
        if window.size == 0:
            return {"error": "no depth reading at that pixel"}
        point = self.models[camera].deproject(u, v, float(np.median(window)), frame["pose"])
        return {"point": [round(float(c), 4) for c in point]}

    def self_check(self, blocks: dict[str, RigidPrim]) -> dict:
        """Project each block's top-face center into both cameras and locate it again (debug only)."""
        errors = {}
        for camera, frame in self.frames.items():
            for color, center in block_poses(blocks).items():
                top = center + [0.0, 0.0, BLOCK_SIZE / 2]
                u, v, _ = self.models[camera].project(top, frame["pose"])
                found = self.locate(camera, u, v)
                if "point" in found:
                    errors[f"{camera}/{color}"] = round(float(np.linalg.norm(np.array(found["point"]) - top)), 4)
        return errors


class LLMClient:
    def __init__(self):
        self.url = args.gateway.rstrip("/") + "/chat/completions"

    def token(self) -> str:
        return os.environ.get("LCA_TOKEN") or open(args.token_file).read().strip()

    def act(self, text: str, images: list[str], model: str | None = None, vision: bool = False) -> tuple[dict, dict]:
        content = [{"type": "text", "text": text}]
        content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b, "detail": "high"}}
                    for b in images if b]
        system = VISION_SYSTEM_PROMPT.format(width=args.width, height=args.height) if vision else SYSTEM_PROMPT
        schema = VISION_ACTION_SCHEMA if vision else ACTION_SCHEMA
        body = {"model": model or args.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": schema}}
        last_error = None
        for attempt in range(5):
            request = urllib.request.Request(self.url, json.dumps(body).encode(),
                                             {"Authorization": "Bearer " + self.token(),
                                              "Content-Type": "application/json"})
            start = time.time()
            try:
                response = json.load(urllib.request.urlopen(request, timeout=180))
                message = response["choices"][0]["message"]["content"]
                usage = response.get("usage") or {}
                meta = {"latency_s": round(time.time() - start, 2),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens")}
                return json.loads(message), meta
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.read()[:300]!r}"
                if exc.code in (400, 401, 403):
                    break
            except Exception as exc:  # network errors and malformed JSON are retried
                last_error = repr(exc)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_error}")


def describe_state(robot: Robot, blocks: dict[str, RigidPrim]) -> str:
    grasp = robot.grasp_position()
    fingers = robot.joint_positions()[7:]
    lines = [f"Gripper grasp point: x={grasp[0]:.3f} y={grasp[1]:.3f} z={grasp[2]:.3f}",
             f"Gripper: commanded {robot.gripper_state.name.lower()}, finger positions "
             f"{fingers[0]:.3f} m and {fingers[1]:.3f} m (0.04 = fully open, 0 = fully closed)"]
    for color, p in block_poses(blocks).items():
        lines.append(f"{color} block center: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}")
    return "\n".join(lines)


def describe_proprioception(robot: Robot) -> str:
    grasp = robot.grasp_position()
    fingers = robot.joint_positions()[7:]
    return (f"Gripper grasp point: x={grasp[0]:.3f} y={grasp[1]:.3f} z={grasp[2]:.3f}\n"
            f"Gripper: commanded {robot.gripper_state.name.lower()}, finger positions "
            f"{fingers[0]:.3f} m and {fingers[1]:.3f} m (0.04 = fully open, 0 = fully closed)")


def run_llm_episode(robot: Robot, blocks: dict[str, RigidPrim], task: tuple[str, str], io: RosIO,
                    client: LLMClient, instruction: str | None = None, model: str | None = None,
                    progress=None, observation: str = "state", perception: Perception | None = None) -> list[dict]:
    instruction = instruction or f"Stack the {task[0]} block on top of the {task[1]} block."
    vision = observation == "vision"
    history: list[str] = []
    steps = []
    for index in range(args.max_steps):
        if vision:
            perception.snapshot(robot)
            text = (f"Task: {instruction}\n\nRobot proprioception:\n{describe_proprioception(robot)}\n\n"
                    f"Previous actions ({len(history)}):\n" + ("\n".join(history) if history else "none") +
                    f"\n\nImages ({args.width}x{args.height}): the scene camera, then the wrist camera.")
            images = perception.images(index)
        else:
            text = (f"Task: {instruction}\n\nMeasured state:\n{describe_state(robot, blocks)}\n\n"
                    f"Previous actions ({len(history)}):\n" + ("\n".join(history) if history else "none") +
                    "\n\nImages: a fixed scene camera looking at the robot from the front, then the wrist camera.")
            images = [encode_jpeg(io.latest_rgb("scene_camera")), encode_jpeg(io.latest_rgb("wrist_camera"))]
        if progress:
            progress({"type": "thinking", "index": index, "sim_time": round(SimulationManager.get_simulation_time(), 3)})
        try:
            decision, meta = client.act(text, images, model=model, vision=vision)
        except RuntimeError as exc:
            steps.append({"index": index, "error": str(exc)})
            print(f"[llm] {exc}", flush=True)
            if progress:
                progress({"type": "error", "index": index, "error": str(exc)})
            break
        if progress:
            progress({"type": "decision", "index": index, "decision": decision, "llm": meta})
        action = decision.get("action")
        target = np.array([decision.get("x", 0.0), decision.get("y", 0.0), decision.get("z", 0.0)], dtype=np.float64)
        debug = None
        if action == "locate" and vision:
            camera, u, v = decision.get("camera", "scene"), float(decision.get("u", 0)), float(decision.get("v", 0))
            result = {**perception.locate(camera, u, v), "duration_s": 0.0}
            summary = f"locate({camera}, {u:.0f}, {v:.0f})"
            if "point" in result:
                # ground truth for analysis only; never shown to the LLM
                tops = {c: p + [0.0, 0.0, BLOCK_SIZE / 2] for c, p in block_poses(blocks).items()}
                nearest = min(tops, key=lambda c: np.linalg.norm(tops[c] - result["point"]))
                debug = {"nearest_block": nearest,
                         "xy_error_to_top_center_m": round(float(np.linalg.norm(tops[nearest][:2] -
                                                                                np.array(result["point"][:2]))), 4)}
        elif action == "move_to":
            clipped = np.array([np.clip(target[i], *MOVE_LIMITS[k]) for i, k in enumerate("xyz")])
            result = robot.move_to(clipped)
            if not np.allclose(clipped, target):
                result["clipped_target"] = [round(float(v), 3) for v in clipped]
            summary = f"move_to({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
        elif action == "open_gripper":
            result = robot.actuate(Gripper.OPEN)
            summary = "open_gripper()"
        elif action == "close_gripper":
            result = robot.actuate(Gripper.CLOSE)
            summary = "close_gripper()"
        else:
            result = {}
            summary = "done()"
        step = {"index": index, "sim_time": round(SimulationManager.get_simulation_time(), 3),
                "prompt": text, "decision": decision, "llm": meta, "result": result}
        if debug:
            step["debug"] = debug
        steps.append(step)
        io.pub_step.publish(String(data=json.dumps({k: v for k, v in step.items() if k != "prompt"})))
        history.append(f"{index + 1}. {summary} -> {json.dumps(result)}")
        print(f"[llm] step {index} {summary} {json.dumps(result)} ({meta['latency_s']} s) "
              f"{decision.get('reasoning', '')[:120]}", flush=True)
        if progress:
            progress({"type": "result", "index": index, "summary": summary, "result": result,
                      "sim_time": step["sim_time"]})
        if action == "done":
            break
    return steps


def run_policy_episode(robot: Robot, blocks: dict[str, RigidPrim], task: tuple[str, str], io: RosIO,
                       connection, progress=None) -> list[dict]:
    """Query the policy at the camera rate and execute the returned chunk.

    Joint policies set joint position targets directly. Goal and end-effector policies return grasp-point goals
    or hand positions plus a gripper open command (> 0.5 = open), which RMPflow and the gripper controllers track.
    """
    task_index = TASKS.index(task)
    substeps = max(1, round(1.0 / (args.camera_hz * PHYSICS_DT)))
    dof_indices = [robot.scenario.articulation.dof_names.index(n) for n in JOINT_NAMES]
    steps = []
    start = robot.t
    while robot.t - start < args.policy_seconds:
        observation = {"scene_rgb": io.latest_rgb("scene_camera"), "wrist_rgb": io.latest_rgb("wrist_camera"),
                       "joint_positions": robot.joint_positions(), "task_index": task_index,
                       "block_positions": np.stack([p for p in block_poses(blocks).values()])}
        connection.send(observation)
        reply = connection.recv()
        if "ee_targets" in reply or "goal_targets" in reply:
            is_goal = "goal_targets" in reply
            targets = np.asarray(reply["goal_targets" if is_goal else "ee_targets"], dtype=np.float64)
            for target in targets:
                grasp_target = target[:3] if is_goal else target[:3] - robot.grasp_offset
                robot.target = np.clip(grasp_target, [MOVE_LIMITS[k][0] for k in "xyz"],
                                       [MOVE_LIMITS[k][1] for k in "xyz"])
                gripper = Gripper.OPEN if target[3] > 0.5 else Gripper.CLOSE
                if gripper != robot.gripper_state:
                    robot.gripper_state = gripper
                    robot.controller.set_next_controller(gripper)
                for _ in range(substeps):
                    robot.control_step()
        else:
            targets = np.asarray(reply["joint_targets"], dtype=np.float64)
            for target in targets:
                for _ in range(substeps):
                    robot.scenario.articulation.set_dof_position_targets(
                        target.reshape(1, -1).astype(np.float32), dof_indices=dof_indices)
                    robot.command = target.copy()
                    robot.step()
        step = {"sim_time": round(SimulationManager.get_simulation_time(), 3), "executed": len(targets),
                "first_target": [round(float(v), 4) for v in targets[0]]}
        steps.append(step)
        if progress and len(steps) % 4 == 0:
            progress({"type": "policy", **step, "elapsed_s": round(robot.t - start, 2)})
    return steps


def reset_scene(scenario: ManipulationScenario, robot: Robot, blocks: dict[str, RigidPrim],
                rng: np.random.Generator) -> None:
    scenario.articulation.reset_to_default_state()
    default = scenario.articulation.get_default_state()[4]
    scenario.articulation.set_dof_position_targets(default)
    randomize_blocks(blocks, rng)
    for _ in range(30):
        simulation_app.update()
    robot.reset()


def run_episode(scenario: ManipulationScenario, robot: Robot, blocks: dict[str, RigidPrim], io: RosIO, *,
                name: str, directory: str, seed: int, task: tuple[str, str], mode: str, client=None,
                connection=None, instruction: str | None = None, model: str | None = None,
                progress=None, observation: str = "state", perception: Perception | None = None,
                table: str | None = None) -> dict:
    """Reset to the layout for ``seed``, run one LLM or policy episode, record it and return the result."""
    table = table or args.table
    set_table(table)
    rng = np.random.default_rng(seed)
    rng.integers(len(TASKS))  # keep the layout identical to the batch run that drew the task from this seed
    reset_scene(scenario, robot, blocks, rng)
    for _ in range(15):
        robot.control_step()
    initial = block_poses(blocks)
    instruction = instruction or f"Stack the {task[0]} block on top of the {task[1]} block."

    recorder = None if args.no_record else Recorder(directory)
    io.publish_latched()
    io.pub_task.publish(String(data=json.dumps({"instruction": instruction, "top": task[0], "bottom": task[1],
                                                "task_index": TASKS.index(task), "seed": seed})))
    start_wall, start_sim = time.time(), SimulationManager.get_simulation_time()
    print(f"[episode] {name}: {instruction}", flush=True)
    error = None
    steps = []
    camera_check = None
    # ros2 bag record refuses an existing output directory, so the LLM's images are moved in afterwards
    image_staging = directory.rstrip("/") + ".llm_images"
    if perception is not None:
        perception.image_dir = image_staging if mode == "llm" and observation == "vision" else None
    try:
        robot.hold(0.5)
        if mode == "llm" and observation == "vision":
            perception.snapshot(robot)
            camera_check = perception.self_check(blocks)
            print(f"[episode] locate self-check (m): {camera_check}", flush=True)
        if mode == "llm":
            steps = run_llm_episode(robot, blocks, task, io, client, instruction=instruction, model=model,
                                    progress=progress, observation=observation, perception=perception)
        else:
            steps = run_policy_episode(robot, blocks, task, io, connection, progress=progress)
        robot.hold(SETTLE_SECONDS)
    except Exception as exc:  # keep going after controller or simulation errors
        error = repr(exc)
        print(f"[episode] {name} error: {error}", flush=True)
    result = evaluate(blocks, task, initial)
    result.update({"episode": name, "seed": seed, "mode": mode, "model": model or args.model,
                   "observation": observation if mode == "llm" else None, "locate_self_check_m": camera_check,
                   "table": table,
                   "instruction": instruction,
                   "task": {"top": task[0], "bottom": task[1], "task_index": TASKS.index(task)},
                   "initial_block_positions": {c: [round(float(v), 4) for v in p] for c, p in initial.items()},
                   "num_steps": len(steps), "error": error,
                   "sim_seconds": round(SimulationManager.get_simulation_time() - start_sim, 3),
                   "wall_seconds": round(time.time() - start_wall, 1)})
    io.pub_result.publish(String(data=json.dumps(result)))
    for _ in range(10):
        rclpy.spin_once(io.node, timeout_sec=0.05)
    if recorder is not None:
        time.sleep(1.0)
        recorder.stop()
    os.makedirs(directory, exist_ok=True)
    if os.path.isdir(image_staging):
        os.replace(image_staging, os.path.join(directory, "llm_images"))
    with open(os.path.join(directory, "episode.json"), "w") as f:
        json.dump({**result, "steps": steps}, f, indent=1)
    print(f"[episode] {name} success={result['success']} xy_error={result['xy_error_m']} "
          f"steps={len(steps)} sim={result['sim_seconds']}s wall={result['wall_seconds']}s", flush=True)
    return result


def write_json(path: str, data: dict) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)


def serve_jobs(scenario: ManipulationScenario, robot: Robot, blocks: dict[str, RigidPrim], io: RosIO,
               perception: Perception) -> None:
    """Run episodes requested as JSON files in ``<jobs>/queue`` (used by the Franka LLM Lab web app).

    Job fields: id, mode (llm | policy), top, bottom, seed, and optionally instruction, model, max_steps,
    policy_address, policy_seconds. Episodes are written to ``<jobs>/runs/<id>``; status and live camera
    frames go to ``<jobs>/status/<id>.json`` and ``<jobs>/live/<id>/{scene,wrist}.jpg``.
    """
    for sub in ("queue", "runs", "status", "live"):
        os.makedirs(os.path.join(args.jobs, sub), exist_ok=True)
    write_json(os.path.join(args.jobs, "worker.json"), {"pid": os.getpid(), "started": time.time(), "gpu": args.gpu})
    print(f"[jobs] waiting for jobs in {args.jobs}/queue", flush=True)
    while simulation_app.is_running():
        queued = sorted(f for f in os.listdir(os.path.join(args.jobs, "queue")) if f.endswith(".json"))
        if not queued:
            simulation_app.update()
            time.sleep(0.2)
            continue
        path = os.path.join(args.jobs, "queue", queued[0])
        with open(path) as f:
            job = json.load(f)
        os.remove(path)
        job_id = job["id"]
        status_path = os.path.join(args.jobs, "status", f"{job_id}.json")
        live_dir = os.path.join(args.jobs, "live", job_id)
        os.makedirs(live_dir, exist_ok=True)
        status = {"id": job_id, "state": "running", "job": job, "started": time.time(), "events": []}

        def progress(event: dict) -> None:
            status["events"].append(event)
            write_json(status_path, status)

        write_json(status_path, status)
        robot.live_dir = live_dir
        connection = None
        try:
            if job["mode"] == "policy":
                host, port = job["policy_address"].split(":")
                connection = Client((host, int(port)), authkey=b"franka-policy")
            observation = job.get("observation") or "state"
            args.max_steps = int(job.get("max_steps") or (40 if observation == "vision" else 24))
            args.policy_seconds = float(job.get("policy_seconds", 30.0))
            result = run_episode(scenario, robot, blocks, io, name=job_id,
                                 directory=os.path.join(args.jobs, "runs", job_id), seed=int(job["seed"]),
                                 task=(job["top"], job["bottom"]), mode=job["mode"],
                                 client=LLMClient() if job["mode"] == "llm" else None, connection=connection,
                                 instruction=job.get("instruction") or None, model=job.get("model"),
                                 progress=progress, observation=observation, perception=perception,
                                 table=job.get("table") or args.table)
            status.update({"state": "done", "result": result})
        except Exception as exc:
            status.update({"state": "error", "error": repr(exc)})
            print(f"[jobs] {job_id} failed: {exc!r}", flush=True)
        finally:
            robot.live_dir = None
            if connection is not None:
                connection.close()
        status["finished"] = time.time()
        write_json(status_path, status)


def main() -> None:
    if not args.jobs and not args.out:
        raise SystemExit("--out is required unless --jobs is given")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device="cpu")
    scenario, blocks = build_scene()
    scenario.initialize_world_binding(exclude_prim_paths=(*(block_path(c) for c in COLORS), TABLE_PATH))
    build_state_graph()
    build_camera_graph("scene_camera", SCENE_CAMERA_PATH)
    build_camera_graph("wrist_camera", WRIST_CAMERA_PATH)

    rclpy.init()
    urdf_files = sorted(load_cumotion_supported_robot("franka").directory.glob("*.urdf"))
    io = RosIO(scenario, blocks, urdf_files[0].read_text() if urdf_files else "")
    simulation_app.update()
    app_utils.play()
    simulation_app.update()
    robot = Robot(scenario, io)
    perception = Perception(io)

    try:
        if args.jobs:
            serve_jobs(scenario, robot, blocks, io, perception)
            return
        client = LLMClient() if args.mode == "llm" else None
        connection = None
        if args.mode == "policy":
            host, port = args.policy_address.split(":")
            connection = Client((host, int(port)), authkey=b"franka-policy")
        for episode in range(args.episodes):
            seed = args.seed + episode
            rng = np.random.default_rng(seed)
            task = TASKS[int(rng.integers(len(TASKS)))]
            prefix = "vision" if args.mode == "llm" and args.observation == "vision" else args.mode
            name = f"{prefix}_seed{seed:06d}"
            directory = os.path.join(args.out, name)
            if os.path.exists(os.path.join(directory, "episode.json")):
                continue
            run_episode(scenario, robot, blocks, io, name=name, directory=directory, seed=seed, task=task,
                        mode=args.mode, client=client, connection=connection, observation=args.observation,
                        perception=perception)
    finally:
        app_utils.stop()
        simulation_app.update()
        io.close()
        rclpy.shutdown()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    os._exit(exit_code)
