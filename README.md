# Winter Project — Perceptual Awareness With Semantics (PAWS)

A ROS2 system for autonomous navigation and semantic object mapping on the Unitree Go2 quadruped. The robot builds a persistent 3D concept map using a GPU-hosted Vision-Language Model (VLM) server, and can navigate to queried objects by name.

**Packages:**
| Package | Purpose |
|---|---|
| `concept_graph` | Semantic object mapping node |
| `concept_graph_interfaces` | `QueryObject` service definition |
| `go2_explore` | Autonomous frontier exploration |
| `vlm_vision` | VLM detection pipeline + goal tracking |
| `unitree/go2_control` | Go2 drivers, SLAM/Nav2 launch files |
| `unitree/go2_description` | URDF/xacro robot model |
| `unitree/unitree_ros2` | CycloneDDS + Unitree message definitions |

## Overview

<img width="3400" height="2680" alt="Winter Project Architecture" src="https://github.com/user-attachments/assets/33885dfc-0c04-4b64-ad81-f52349eaa74c" />


## 1. Prerequisites

Install ROS2 Kilted and required packages:

```bash
sudo apt install \
  ros-kilted-rtabmap-ros \
  ros-kilted-nav2-bringup \
  ros-kilted-rmw-cyclonedds-cpp \
  ros-kilted-rosidl-generator-dds-idl \
  ros-kilted-realsense2-camera
```


Python dependencies (for VLM nodes):

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy requests opencv-python
```

---

## 2. Unitree ROS2 Setup

The Unitree Go2 communicates over CycloneDDS.

### 2a. Build the CycloneDDS workspace

The `cyclonedds_ws` inside `unitree_ros2` must be built **before** sourcing any ROS2 environment:

```bash
# If /opt/ros/kilted/setup.bash is in your ~/.bashrc, comment it out first,
# then open a fresh terminal before running the following.

cd ~/WinterProject/ws/src/unitree/unitree_ros2/cyclonedds_ws
colcon build --packages-select cyclonedds
source /opt/ros/kilted/setup.bash
colcon build   # builds unitree_go and unitree_api
```

### 2b. Configure the network interface

pen `unitree/unitree_ros2/setup.sh` and replace the `NetworkInterface name` with your interface:

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="enp0s31f6" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'
```

Also set the IP of the Ethernet interface to `192.168.123.99` (mask `255.255.255.0`) in your network settings.

### 2c. Source the Unitree environment

Every terminal that interacts with the Go2 must source this script, before any other ROS2 sourcing:

```bash
source ~/WinterProject/ws/src/unitree/unitree_ros2/setup.sh
```

---

## 3. VLM Inference Server Setup

The VLM inference server runs on a separate GPU workstation. The source lives at:

```
/home/lea1212/WinterProject/ws/src/winter_project/vlm_service/
```

On the workstation, start the FastAPI server:

```bash
cd /path/to/vlm_service
uvicorn server_app:app --host 0.0.0.0 --port 8000
```

Then port-forward from the robot's computer so the ROS nodes can reach it at `http://127.0.0.1:8000`:

```bash
ssh -L 8000:localhost:8000 <user>@<workstation-ip>
```

