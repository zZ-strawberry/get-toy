#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>

using namespace cv;
using namespace std;

// 计算轮廓中心点
Point2f getContourCenter(const vector<Point>& contour) {
    Moments m = moments(contour);
    if (m.m00 != 0) {
        return Point2f(m.m10 / m.m00, m.m01 / m.m00);
    }
    return Point2f(0, 0);
}

class BlackObjectDetectionNode : public rclcpp::Node {
public:
    BlackObjectDetectionNode() : Node("black_object_detection_node") {
        // 创建发布器
        status_pub_ = this->create_publisher<std_msgs::msg::Float32>("black_object/status", 10);  // 0.00 或 1.00
        angle_pub_  = this->create_publisher<std_msgs::msg::Float32>("black_object/angle", 10);   // 弧度 两位小数
        offset_pub_ = this->create_publisher<std_msgs::msg::Float32>("black_object/offset", 10);  // Y偏移 两位小数
        
        // 使用定时器处理帧
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&BlackObjectDetectionNode::processFrame, this));
        
        // 打开摄像头
        cap_.open(0);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开摄像头");
            rclcpp::shutdown();
            return;
        }

        // 仅保留主显示窗口
        cv::namedWindow("Black Object Detection", WINDOW_AUTOSIZE);

        RCLCPP_INFO(this->get_logger(), "黑色物体检测节点已启动 (HSV模式)");
    }
    
    ~BlackObjectDetectionNode() {
        cap_.release();
        destroyAllWindows();
    }

private:
    void processFrame() {
        Mat frame, hsv, binary, processed;
        
        cap_ >> frame;
        if (frame.empty()) {
            return;
        }
        
        // 转换到HSV色彩空间并预处理
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        int blur = blur_size_ * 2 + 1;
        GaussianBlur(hsv, hsv, Size(blur, blur), 0);

        // HSV黑色检测（H全范围，V低、S低）
        Scalar lower_black(0, 0, 0);
        Scalar upper_black(180, s_max_, v_max_);
        inRange(hsv, lower_black, upper_black, binary);

        // 形态学处理
        int morph = morph_size_ + 1;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(morph, morph));
        morphologyEx(binary, processed, MORPH_CLOSE, kernel);
        morphologyEx(processed, processed, MORPH_OPEN, kernel);

        // 查找最大轮廓
        vector<vector<Point>> contours;
        findContours(processed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        bool found = false;
        float angle = 0.0f;
        int32_t y_offset_px = 0; 
        Point2f center(0, 0);
        double maxArea = 0.0;
        vector<Point> maxContour;

        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < min_area_ * 100) continue;
            if (area > maxArea) {
                maxArea = area;
                maxContour = contour;
                found = true;
            }
        }

        // 画面中心
        Point2f imageCenter(frame.cols / 2.0f, frame.rows / 2.0f);

        if (found) {
            // 计算中心点与弧度角：以竖直轴上半轴为起始，右侧为正、左侧为负
            center = getContourCenter(maxContour);
            float dx = center.x - imageCenter.x;
            float dy = imageCenter.y - center.y; // 上为正
            angle = std::atan2(dx, dy);
            y_offset_px = static_cast<int32_t>(std::lround(center.y - imageCenter.y)); // 取整像素偏移

            // 可视化
            drawContours(frame, vector<vector<Point>>{maxContour}, -1, Scalar(0, 255, 0), 2);
            Rect boundingBox = boundingRect(maxContour);
            rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);
            circle(frame, center, 8, Scalar(0, 0, 255), -1);
            line(frame, imageCenter, center, Scalar(255, 255, 0), 2);

            char text[100];
            sprintf(text, "Angle: %.2f rad", angle);
            putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        // 发布状态、弧度、y轴偏移
        publishData(found, angle, y_offset_px);

        // 十字线与显示
        circle(frame, imageCenter, 5, Scalar(255, 0, 0), -1);
        line(frame, Point(imageCenter.x, 0), Point(imageCenter.x, frame.rows), Scalar(255, 0, 0), 1);
        line(frame, Point(0, imageCenter.y), Point(frame.cols, imageCenter.y), Scalar(255, 0, 0), 1);

        imshow("Black Object Detection", frame);
        waitKey(1);
    }
    
    void publishData(bool hasTarget, float angle, int32_t yOffsetPx) {
        auto status_msg = std_msgs::msg::Float32();
        float status_val = hasTarget ? 1.0f : 0.0f;
        status_msg.data = std::round(status_val * 100.0f) / 100.0f; // 0.00 或 1.00
        status_pub_->publish(status_msg);

        auto angle_msg = std_msgs::msg::Float32();
        angle_msg.data = std::round(angle * 100.0f) / 100.0f; // 弧度
        angle_pub_->publish(angle_msg);

        auto offset_msg = std_msgs::msg::Float32();
        float offset_val = static_cast<float>(yOffsetPx);
        offset_msg.data = std::round(offset_val * 100.0f) / 100.0f; // 像素偏移
        offset_pub_->publish(offset_msg);
    }
    
    VideoCapture cap_;
    
    int v_max_ = 58;     // V通道上限（暗）
    int s_max_ = 255;    // S通道上限
    int blur_size_ = 2;  // 实际大小 = blur_size_ * 2 + 1
    int min_area_ = 10;  // 实际面积 = min_area_ * 100
    int morph_size_ = 3; // 形态学核大小
    
    // 发布器类型全部改为 Float32
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr status_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr angle_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_pub_; 
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BlackObjectDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
