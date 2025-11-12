from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='color_tracking_node',
            executable='basket',
            name='black_object_detection_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                # 可以在这里添加参数配置
                # 'v_max': 58,
                # 'absolute_min_area': 2000.0,
            }]
        )
    ])
