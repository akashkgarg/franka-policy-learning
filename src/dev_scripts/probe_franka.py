from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0})

import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)

import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import ManipulationScenario

SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
scenario = ManipulationScenario("franka", robot_prim_path="/World/robot")
scenario.setup_scene()
stage = omni.usd.get_context().get_stage()
with open("/tmp/probe_franka.txt", "w") as f:
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.count("/") <= 4:
            f.write(f"{path} {prim.GetTypeName()} {[s for s in prim.GetAppliedSchemas()][:4]}\n")
    f.write(f"config: {scenario.robot_config}\n")
    f.write(f"joint_space: {scenario.joint_space}\n")
    f.write(f"site_space: {scenario.site_space}\n")
    robot = load_cumotion_supported_robot("franka")
    f.write(f"cumotion robot attrs: {[a for a in dir(robot) if not a.startswith('_')]}\n")
    f.write(f"cumotion robot: {robot}\n")
simulation_app.close()
