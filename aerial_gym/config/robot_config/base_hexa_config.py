import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.camera_config.base_normal_faceID_camera_config import (
    BaseNormalFaceIDCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class BaseHexaCfg:

    class init_config:
        min_init_state = [
            0.1,
            0.15,
            0.15,
            0,
            0,
            -np.pi / 6,
            1.0,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
        ]
        max_init_state = [
            0.2,
            0.85,
            0.85,
            0,
            0,
            np.pi / 6,
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]

    class sensor_config:
        enable_camera = False
        camera_config = BaseDepthCameraConfig
        enable_lidar = False
        lidar_config = BaseLidarConfig
        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:
        enable_disturbance = False
        prob_apply_disturbance = 0.02
        max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/hexa"
        file = "hexa2.urdf"
        name = "base_hexaroter"
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        collision_mask = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

        min_state_ratio = [
            0.1,
            0.1,
            0.1,
            0,
            0,
            -np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        max_state_ratio = [
            0.3,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        max_force_and_torque_disturbance = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]

        color = None
        semantic_masked_links = {}
        keep_in_env = True

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        use_collision_mesh_instead_of_visual = False

    class control_allocator_config:
        num_motors = 6
        force_application_level = "motor_link"
        application_mask = [1 + 4 + i for i in range(6)]
        motor_directions = [1, -1, 1, -1, 1, -1]
        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # fx
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # fy
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # fz
            [-0.13, -0.13, 0.13, 0.13, 0.17, -0.17],  # tx
            [-0.13, 0.13, 0.13, -0.13, 0.0, 0.0],      # ty
            [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],   # tz
        ]

        class motor_model_config:
            use_rps = False
            motor_thrust_constant_min = 0.00000926312
            motor_thrust_constant_max = 0.00001826312
            motor_time_constant_increasing_min = 0.03
            motor_time_constant_increasing_max = 0.03
            motor_time_constant_decreasing_min = 0.04
            motor_time_constant_decreasing_max = 0.04
            max_thrust = 2
            min_thrust = 0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.01
            use_discrete_approximation = True


class BaseHexaWithImuCfg(BaseHexaCfg):
    class sensor_config(BaseHexaCfg.sensor_config):
        enable_imu = True
        imu_config = BaseImuConfig


class BaseHexaWithCameraCfg(BaseHexaCfg):
    class sensor_config(BaseHexaCfg.sensor_config):
        enable_camera = True
        camera_config = BaseDepthCameraConfig


class BaseHexaWithCameraImuCfg(BaseHexaCfg):
    class sensor_config(BaseHexaCfg.sensor_config):
        enable_camera = True
        camera_config = BaseDepthCameraConfig
        enable_imu = True
        imu_config = BaseImuConfig


class BaseHexaWithLidarCfg(BaseHexaCfg):
    class sensor_config(BaseHexaCfg.sensor_config):
        enable_lidar = True
        lidar_config = BaseLidarConfig


class BaseHexaWithFaceIDNormalCameraCfg(BaseHexaCfg):
    class sensor_config(BaseHexaCfg.sensor_config):
        enable_camera = True
        camera_config = BaseNormalFaceIDCameraConfig
