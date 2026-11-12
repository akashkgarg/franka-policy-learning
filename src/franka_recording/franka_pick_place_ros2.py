"""Franka pick-and-place that publishes every available sensor and state stream over ROS 2.

Run through ``record_franka_bag.sh`` to capture the topics into a ROS 2 bag with MCAP storage.

Published topics (all stamped with simulation time):
    /clock                                   rosgraph_msgs/Clock
    /joint_states                            sensor_msgs/JointState (position, velocity, effort)
    /joint_commands                          sensor_msgs/JointState (controller position/velocity targets)
    /tf, /tf_static                          tf2_msgs/TFMessage (robot links, cube, place target, cameras)
    /robot_description                       std_msgs/String (URDF, transient local)
    /ee_pose                                 geometry_msgs/PoseStamped (tool frame)
    /cube/pose, /cube/twist                  geometry_msgs/PoseStamped, TwistStamped
    /hand_imu                                sensor_msgs/Imu
    /left_finger/contact, /right_finger/contact  geometry_msgs/WrenchStamped (net contact force, world frame)
    /pick_place/phase                        std_msgs/String
    /<camera>/rgb                            sensor_msgs/Image
    /<camera>/depth                          sensor_msgs/Image (32FC1, meters)
    /<camera>/points                         sensor_msgs/PointCloud2
    /<camera>/instance_segmentation          sensor_msgs/Image (+ /<camera>/instance_labels)
    /<camera>/semantic_segmentation          sensor_msgs/Image (+ /<camera>/semantic_labels)
    /<camera>/bbox_2d                        vision_msgs/Detection2DArray
    /<camera>/bbox_3d                        vision_msgs/Detection3DArray
    /<camera>/camera_info                    sensor_msgs/CameraInfo
where <camera> is scene_camera and wrist_camera.
"""

from __future__ import annotations

import argparse
import math

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--width", type=int, default=640, help="Camera image width")
parser.add_argument("--height", type=int, default=480, help="Camera image height")
parser.add_argument("--camera-hz", type=float, default=15.0, help="Image and point cloud publish rate (sim time)")
parser.add_argument("--preroll", type=float, default=2.0, help="Seconds to publish before the arm starts moving")
parser.add_argument("--postroll", type=float, default=1.0, help="Seconds to publish after placement settles")
parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "multi_gpu": False, "active_gpu": 0})

import isaacsim.core.experimental.utils.app as app_utils

app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.robot_motion.examples")
simulation_app.update()

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.graph.core as og
import omni.usd
import rclpy
import usdrt.Sdf
import warp as wp
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped, WrenchStamped
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.utils.semantics import add_labels
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import (
    GripperCommand,
    JointGripperController,
    ManipulationScenario,
    PickPlaceController,
)
from isaacsim.sensors.experimental.physics import IMU, Contact, ContactSensor, IMUSensor
from isaacsim.sensors.experimental.rtx import RtxCamera
from pxr import Gf, UsdGeom
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

PHYSICS_DT = 1.0 / 60.0
WORLD_FRAME = "World"
ROBOT_PATH = "/World/robot"
CUBE_PATH = "/World/Cube"
TARGET_PATH = "/World/PlaceTarget"
HAND_PATH = f"{ROBOT_PATH}/panda_hand"
SCENE_CAMERA_PATH = "/World/scene_camera"
WRIST_CAMERA_PATH = f"{HAND_PATH}/wrist_camera"
PICK = (0.45, 0.0, 0.025)
PLACE = (0.45, 0.35, 0.025)
PLACE_STACK_HEIGHT = 0.055 + 0.025 + 0.004
PLACE_STALL_SECONDS = 0.25
PLACE_STALL_DISTANCE = 0.001
PLACE_STALL_POSITION_TOLERANCE = 0.10
PLACE_STALL_ORIENTATION_TOLERANCE = 0.5


def sim_stamp(seconds: float) -> Time:
    sec = int(math.floor(seconds))
    return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)) % 1_000_000_000)


def define_camera(path: str, focal_length: float = 18.0) -> UsdGeom.Camera:
    camera = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), path)
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetVerticalApertureAttr().Set(20.955 * args.height / args.width)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return camera


def build_scene() -> tuple[ManipulationScenario, RigidPrim]:
    scenario = ManipulationScenario("franka", robot_prim_path=ROBOT_PATH)
    scenario.setup_scene()
    cube = Cube(CUBE_PATH, positions=[PICK], sizes=0.05, colors=[[0.8, 0.1, 0.1]])
    cube_prim = RigidPrim(cube.paths)
    GeomPrim(cube.paths, apply_collision_apis=True)
    target = Cube(TARGET_PATH, positions=[PLACE], sizes=0.055, colors=[[0.1, 0.8, 0.1]])
    GeomPrim(target.paths, apply_collision_apis=True)

    for path, label in ((ROBOT_PATH, "franka"), (CUBE_PATH, "cube"), (TARGET_PATH, "place_target"),
                        ("/World/ground_plane", "ground")):
        add_labels(path, labels=label)

    scene_camera = define_camera(SCENE_CAMERA_PATH)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.5, 1.1, 0.9), Gf.Vec3d(0.4, 0.15, 0.1), Gf.Vec3d(0, 0, 1))
    scene_camera.AddTransformOp().Set(view.GetInverse())

    # USD cameras look down -Z; flip so the wrist camera looks along the hand's +Z toward the fingers
    wrist_camera = define_camera(WRIST_CAMERA_PATH, focal_length=12.0)
    xform = UsdGeom.XformCommonAPI(wrist_camera)
    xform.SetTranslate(Gf.Vec3d(0.06, 0.0, 0.02))
    xform.SetRotate(Gf.Vec3f(180.0, 0.0, 90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    # RtxCamera rewrites the xform ops, so the poses above must be authored first
    for path in (SCENE_CAMERA_PATH, WRIST_CAMERA_PATH):
        RtxCamera(path, tick_rate=args.camera_hz)
    return scenario, cube_prim


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
                (
                    "PublishTF.inputs:targetPrims",
                    [usdrt.Sdf.Path(p) for p in (ROBOT_PATH, CUBE_PATH, TARGET_PATH, SCENE_CAMERA_PATH,
                                                 WRIST_CAMERA_PATH)],
                ),
            ],
        },
    )


def build_camera_graph(name: str, camera_path: str) -> None:
    """Publish every camera annotator the ROS 2 camera helper supports for one camera."""
    keys = og.Controller.Keys
    frame_id = f"{name}_optical"
    helpers = [
        ("rgb", "rgb"),
        ("depth", "depth"),
        ("points", "depth_pcl"),
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


class StatePublisher:
    """Publishes controller, object, IMU and contact data that the OmniGraph nodes do not cover."""

    def __init__(self, scenario: ManipulationScenario, cube: RigidPrim, urdf: str):
        self.scenario = scenario
        self.cube = cube
        self.node = rclpy.create_node("franka_sim_state")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_description = self.node.create_publisher(String, "robot_description", latched)
        self.pub_tf_static = self.node.create_publisher(TFMessage, "tf_static", latched)
        self.pub_commands = self.node.create_publisher(JointState, "joint_commands", 10)
        self.pub_ee = self.node.create_publisher(PoseStamped, "ee_pose", 10)
        self.pub_cube_pose = self.node.create_publisher(PoseStamped, "cube/pose", 10)
        self.pub_cube_twist = self.node.create_publisher(TwistStamped, "cube/twist", 10)
        self.pub_imu = self.node.create_publisher(Imu, "hand_imu", 10)
        self.pub_phase = self.node.create_publisher(String, "pick_place/phase", 10)
        self.pub_contacts = {
            side: self.node.create_publisher(WrenchStamped, f"{side}_finger/contact", 10)
            for side in ("left", "right")
        }
        self.imu = IMUSensor(IMU.create(f"{HAND_PATH}/imu_sensor", translations=np.zeros((1, 3))))
        self.contacts = {}
        for side in ("left", "right"):
            sensor = ContactSensor(Contact.create(f"{ROBOT_PATH}/panda_{side}finger/contact_sensor",
                                                  min_threshold=0, max_threshold=1e7, radius=-1,
                                                  translations=np.zeros((1, 3))))
            sensor.add_raw_contact_data_to_frame()
            self.contacts[side] = sensor
        self.pub_description.publish(String(data=urdf))
        self.pub_tf_static.publish(TFMessage(transforms=[self._optical_frame(n)
                                                         for n in ("scene_camera", "wrist_camera")]))

    @staticmethod
    def _optical_frame(camera: str) -> TransformStamped:
        # USD camera axes (x right, y up, -z forward) to ROS optical axes (x right, y down, z forward)
        t = TransformStamped()
        t.header.frame_id = camera
        t.child_frame_id = f"{camera}_optical"
        t.transform.rotation.x = 1.0
        t.transform.rotation.w = 0.0
        return t

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

    def publish(self, sim_time: float, desired: mg.RobotState | None, phase: str) -> None:
        stamp = sim_stamp(sim_time)

        if desired is not None and desired.joints is not None:
            joints = desired.joints
            msg = JointState()
            msg.header.stamp = stamp
            names = list(joints.position_names) if joints.positions is not None else []
            msg.name = names
            if joints.positions is not None:
                msg.position = [float(v) for v in joints.positions.numpy().reshape(-1)]
            if joints.velocities is not None and list(joints.velocity_names) == names:
                msg.velocity = [float(v) for v in joints.velocities.numpy().reshape(-1)]
            self.pub_commands.publish(msg)

        ee_positions, ee_orientations = self.scenario.read_tool_pose()
        self.pub_ee.publish(self._pose(stamp, np.asarray(ee_positions).reshape(-1, 3)[0],
                                       np.asarray(ee_orientations).reshape(-1, 4)[0]))

        positions, orientations = self.cube.get_world_poses()
        self.pub_cube_pose.publish(self._pose(stamp, positions.numpy()[0], orientations.numpy()[0]))
        linear, angular = self.cube.get_velocities()
        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = WORLD_FRAME
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = (float(v) for v in linear.numpy()[0])
        twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = (float(v) for v in angular.numpy()[0])
        self.pub_cube_twist.publish(twist)

        imu_data = self.imu.get_data()
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "panda_hand"
        w, x, y, z = (float(v) for v in np.asarray(imu_data["orientation"]).reshape(-1)[:4])
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = w, x, y, z
        ax, ay, az = (float(v) for v in np.asarray(imu_data["linear_acceleration"]).reshape(-1)[:3])
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = ax, ay, az
        gx, gy, gz = (float(v) for v in np.asarray(imu_data["angular_velocity"]).reshape(-1)[:3])
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gx, gy, gz
        self.pub_imu.publish(imu)

        for side, sensor in self.contacts.items():
            data = sensor.get_data()
            impulse = sum((c["impulse"] for c in data.get("contacts") or []), np.zeros(3, dtype=np.float32))
            force = impulse / PHYSICS_DT
            wrench = WrenchStamped()
            wrench.header.stamp = stamp
            wrench.header.frame_id = WORLD_FRAME
            wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z = (float(v) for v in force)
            self.pub_contacts[side].publish(wrench)

        self.pub_phase.publish(String(data=phase))

    def close(self) -> None:
        self.node.destroy_node()


def make_controller(scenario: ManipulationScenario) -> tuple[PickPlaceController, str]:
    cumotion_robot = load_cumotion_supported_robot("franka")
    tool_frame = scenario.robot_config.tool.controller_frame
    gripper = scenario.robot_config.gripper
    gripper_args = {
        "robot_joint_space": scenario.joint_space,
        "joint_names": gripper.joint_names,
        "open_positions": gripper.open_positions,
        "closed_positions": gripper.closed_positions,
        "position_tolerance": gripper.position_tolerance,
        "velocity_tolerance": gripper.velocity_tolerance,
        "minimum_close_fraction": gripper.minimum_close_fraction,
    }
    controller = PickPlaceController(
        arm_controller=RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=scenario.world_interface,
            robot_joint_space=scenario.joint_space,
            robot_site_space=scenario.site_space,
            tool_frame=tool_frame,
        ),
        gripper_open_controller=JointGripperController(**gripper_args, command=GripperCommand.OPEN),
        gripper_close_controller=JointGripperController(**gripper_args, command=GripperCommand.CLOSE),
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
        controller_to_grasp_position=scenario.robot_config.tool.controller_to_grasp_position,
        controller_to_grasp_orientation=scenario.robot_config.tool.controller_to_grasp_orientation,
        grasp_orientation=scenario.robot_config.grasp_orientation,
        approach_height=0.20,
    )
    urdf_files = sorted(cumotion_robot.directory.glob("*.urdf"))
    urdf = urdf_files[0].read_text() if urdf_files else ""
    return controller, urdf


def make_goal() -> mg.RobotState:
    names = [PickPlaceController.PICK_SITE, PickPlaceController.PLACE_SITE]
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=names,
            positions=(names, wp.array([PICK, PLACE], dtype=wp.float32, device="cpu")),
        )
    )


def main() -> None:
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device="cpu")
    scenario, cube = build_scene()
    scenario.initialize_world_binding(exclude_prim_paths=(TARGET_PATH,))
    controller, urdf = make_controller(scenario)
    goal = make_goal()

    build_state_graph()
    build_camera_graph("scene_camera", SCENE_CAMERA_PATH)
    build_camera_graph("wrist_camera", WRIST_CAMERA_PATH)

    rclpy.init()
    publisher = StatePublisher(scenario, cube, urdf)
    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    def step(desired: mg.RobotState | None, phase: str) -> None:
        simulation_app.update()
        publisher.publish(SimulationManager.get_simulation_time(), desired, phase)

    try:
        for _ in range(round(args.preroll / PHYSICS_DT)):
            step(None, "PREROLL")

        controller_time = 0.0
        if not controller.reset(scenario.read_robot_state(), goal, t=controller_time):
            raise RuntimeError(controller.failure_reason or "PickPlaceController reset failed.")
        max_steps = math.ceil(sum(controller.phase_timeouts.values()) / PHYSICS_DT)
        steps = 0
        tolerances = (controller._position_tolerance, controller._orientation_tolerance)
        stall_window = round(PLACE_STALL_SECONDS / PHYSICS_DT)
        tool_history = []
        while simulation_app.is_running() and not controller.is_done and not controller.failed:
            controller_time += PHYSICS_DT
            scenario.sync_world()
            previous_phase = controller.phase
            desired = controller.forward(scenario.read_robot_state(), goal, controller_time)
            if desired is None:
                raise RuntimeError(controller.failure_reason or "PickPlaceController returned no command.")
            scenario.apply_robot_state(desired)
            if previous_phase is not controller.phase:
                print(f"[franka] t={controller_time:.2f}s phase {previous_phase.name} -> {controller.phase.name}")
                if controller.phase.name == "DESCEND_PICK":
                    scenario.set_planning_obstacles_enabled((CUBE_PATH,), False)
                elif controller.phase.name == "RETREAT":
                    scenario.set_planning_obstacles_enabled((CUBE_PATH,), True)
                if previous_phase.name == "RELEASE":
                    controller._position_tolerance, controller._orientation_tolerance = tolerances
                tool_history.clear()
            if controller.phase.name == "DESCEND_PLACE" and controller._position_tolerance == tolerances[0]:
                tool_history.append(np.asarray(scenario.read_tool_pose()[0]).reshape(-1, 3)[0].copy())
                cube_z = float(cube.get_world_poses()[0].numpy()[0][2])
                arm_stopped = (len(tool_history) > stall_window and np.linalg.norm(
                    tool_history[-1] - tool_history[-1 - stall_window]) < PLACE_STALL_DISTANCE)
                if cube_z <= PLACE_STACK_HEIGHT or arm_stopped:
                    reason = "cube resting on place target" if cube_z <= PLACE_STACK_HEIGHT else "arm stopped"
                    print(f"[franka] t={controller_time:.2f}s {reason} during DESCEND_PLACE; releasing here")
                    controller._position_tolerance = PLACE_STALL_POSITION_TOLERANCE
                    controller._orientation_tolerance = PLACE_STALL_ORIENTATION_TOLERANCE
            step(desired, controller.phase.name)
            steps += 1
            if steps > max_steps:
                raise TimeoutError("Pick/place did not finish within the controller phase timeouts.")
        if controller.failed:
            raise RuntimeError(controller.failure_reason or "Pick-and-place failed.")

        for _ in range(round(args.postroll / PHYSICS_DT)):
            step(None, "DONE")

        final = cube.get_world_poses()[0].numpy()[0]
        error = float(np.linalg.norm(final[:2] - np.asarray(PLACE[:2])))
        print(f"[franka] pick/place finished; cube XY error from place target {error:.3f} m")
    finally:
        app_utils.stop()
        simulation_app.update()
        publisher.close()
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
        simulation_app.close(exit_code=exit_code)
    if exit_code:
        raise SystemExit(exit_code)
