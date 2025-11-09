from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包路径
    pkg_dir = get_package_share_directory('color_tracking_node')
    
    # 参数文件路径
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    
    # 确保参数文件存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"参数文件不存在: {config_file}")
    
    # 创建节点
    color_tracking_node = Node(
        package='color_tracking_node',
        executable='color_tracking_node',
        name='color_tracking_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('color_tracking/result', '/tracking/result'),
            ('color_tracking/image', '/tracking/image'),
            ('color_tracking/mask', '/tracking/mask')
        ]
    )
    
    return LaunchDescription([
        color_tracking_node
    ])