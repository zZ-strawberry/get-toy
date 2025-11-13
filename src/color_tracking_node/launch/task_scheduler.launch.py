from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    任务调度器launch文件
    
    启动后订阅 /task_command 话题，接收Int32消息:
    - 1: 启动颜色跟踪任务
    - 2: 启动识别放置框任务
    - 0: 停止当前任务
    """
    return LaunchDescription([
        Node(
            package='color_tracking_node',
            executable='task_scheduler',
            name='task_scheduler_node',
            output='screen',
            emulate_tty=True,
        )
    ])
