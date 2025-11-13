/**
 * @file task_scheduler.cpp
 * @brief 任务调度节点，根据订阅的话题消息启动不同的任务
 * 
 * 功能说明:
 * - 订阅 /task_command 话题
 * - 接收到1: 启动颜色跟踪任务
 * - 接收到2: 启动识别放置框任务
 * - 接收到0: 停止当前运行的任务
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <thread>
#include <memory>
#include <atomic>

class TaskScheduler : public rclcpp::Node
{
public:
    TaskScheduler() : Node("task_scheduler"), current_task_(0), task_running_(false)
    {
        // 创建订阅者，订阅任务命令话题
        subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "/task_command", 10,
            std::bind(&TaskScheduler::task_command_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "任务调度节点已启动");
        RCLCPP_INFO(this->get_logger(), "等待任务命令...");
        RCLCPP_INFO(this->get_logger(), "  发送 1 -> 启动颜色跟踪任务");
        RCLCPP_INFO(this->get_logger(), "  发送 2 -> 启动识别放置框任务");
        RCLCPP_INFO(this->get_logger(), "  发送 0 -> 停止当前任务");
    }

    ~TaskScheduler()
    {
        stop_current_task();
    }

private:
    /**
     * @brief 任务命令回调函数
     */
    void task_command_callback(const std_msgs::msg::Int32::SharedPtr msg)
    {
        int command = msg->data;
        RCLCPP_INFO(this->get_logger(), "收到任务命令: %d", command);

        if (command == 0) {
            stop_current_task();
        } else if (command == 1) {
            execute_task(1, "颜色跟踪任务", "color_tracking.launch.py");
        } else if (command == 2) {
            execute_task(2, "识别放置框任务", "basket_detection.launch.py");
        } else {
            RCLCPP_WARN(this->get_logger(), "未知的任务命令: %d", command);
        }
    }

    /**
     * @brief 执行指定的任务
     */
    void execute_task(int task_id, const std::string& task_name, const std::string& launch_file)
    {
        // 如果已有任务在运行，先停止
        if (task_running_) {
            RCLCPP_INFO(this->get_logger(), "停止当前任务...");
            stop_current_task();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // 启动新任务
        current_task_ = task_id;
        task_running_ = true;

        RCLCPP_INFO(this->get_logger(), "启动 %s...", task_name.c_str());

        // 在新线程中启动launch文件
        task_thread_ = std::thread([this, launch_file, task_name]() {
            std::string command = "ros2 launch color_tracking_node " + launch_file;
            RCLCPP_INFO(this->get_logger(), "执行命令: %s", command.c_str());
            
            int result = std::system(command.c_str());
            
            if (result == 0) {
                RCLCPP_INFO(this->get_logger(), "%s 已完成", task_name.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "%s 执行失败，返回码: %d", task_name.c_str(), result);
            }
            
            task_running_ = false;
        });

        task_thread_.detach();
        
        RCLCPP_INFO(this->get_logger(), "%s 已在后台启动", task_name.c_str());
    }

    /**
     * @brief 停止当前运行的任务
     */
    void stop_current_task()
    {
        if (!task_running_) {
            RCLCPP_INFO(this->get_logger(), "当前没有运行中的任务");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "正在停止任务 %d...", current_task_.load());

        // 发送停止命令 (通过系统命令杀死相关进程)
        // 注意：这是一个简单的实现，实际应用中可能需要更优雅的方式
        if (current_task_ == 1) {
            std::system("pkill -f color_tracking_node");
        } else if (current_task_ == 2) {
            std::system("pkill -f basket");
        }

        task_running_ = false;
        current_task_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "任务已停止");
    }

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
    std::thread task_thread_;
    std::atomic<int> current_task_;
    std::atomic<bool> task_running_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TaskScheduler>();
    
    RCLCPP_INFO(node->get_logger(), "任务调度节点已就绪");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}
