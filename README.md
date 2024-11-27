# Legged Robot
## Documents
[Isaac-sim](https://docs.omniverse.nvidia.com/isaacsim/latest/how_to_guides/robots_simulation.html)

[Isaac-lab](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html)

[Isaac-lab中文](https://docs.robotsfan.com/isaaclab/index.html)

```bash
./isaaclab.sh -p source/standalone/RL-project/legged_robot.py 
```



## Install

```shell
conda info -e
conda activate legged_robot
```

https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html#isaac-sim-app-install-workstation/

```shell
sudo apt-get  install libfuse2
```
## External Editors
### Vscode

https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_code_editors.html

Window > Extensions > Searching for `omni.isaac.vscode`

```python
# test
from pxr import Usd, UsdGeom
import omni

stage = omni.usd.get_context().get_stage()
xformPrim = UsdGeom.Xform.Define(stage, '/hello')
spherePrim = UsdGeom.Sphere.Define(stage, '/hello/world')
```
### Jupyter Lab

https://jupyter.org/install

```bash
pip install jupyterlab
jupyter
```
Window > Extensions > Searching for `omni.isaac.jupyter_notebook`

## Import URDF

### SolidWorks to URDF

https://wiki.ros.org/action/fullsearch/sw_urdf_exporter?action=fullsearch&context=180&value=linkto:%22sw_urdf_exporter%22

https://blog.csdn.net/qq_54900679/article/details/137279115

Pont & Axis & Coordinate System



### Import URDF to Isaac Sim

https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/ext_omni_isaac_urdf.html

https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_urdf.html

### Add Joint Drive

Eye >Show By Type > Physics > Colliders > All

## Isaac-lab Install

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#

## Train Franka

https://docs.robotsfan.com/isaaclab/source/tutorials/03_envs/run_rl_training.html

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 64 --headless
```

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 32
```

## Interacting with an articulation
[Writing an Asset Configuration](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html)

## IsaacLab IMU

```bash
/home/tao/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/sensors/imu
```

## Direct RL Env

```python
# spaces
action_space = 6
state_space = 0
observation_space = [tiled_camera.height, tiled_camera.width, 3]

# reset
max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

self.scene.sensors["imu"] = self._imu

def _apply_action(self) -> None:
	self._robot.set_joint_position_target(self.actions, joint_ids=self._cart_dof_idx)
```
```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Robot-Direct --num_envs 64
```

```
import numpy as np

def quaternion_to_angle(quaternion):
    w, x, y, z = quaternion
    
    # 机器人x轴旋转后的方向（旋转矩阵第一列）
    x_prime = 1 - 2 * (y**2 + z**2)
    y_prime = 2 * (x*y + w*z)
    z_prime = 2 * (x*z - w*y)
    
    # 与世界x轴（1, 0, 0）的夹角
    angle = np.arccos(x_prime)  # 得到弧度
    angle_deg = np.degrees(angle)  # 转换为角度
    
    return angle_deg

# 示例四元数
quaternion = (0.707, 0.707, 0, 0)  # 示例：45度旋转
angle = quaternion_to_angle(quaternion)
print(f"机器人x轴相对世界x轴的偏离角度：{angle:.2f}°")
```
