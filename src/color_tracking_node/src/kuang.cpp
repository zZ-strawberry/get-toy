#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp> // 修改状态消息类型为整数

using namespace cv;
using namespace std;

// 判断轮廓是否为正方形
bool isSquare(const vector<Point>& contour, double epsilon = 0.02) {
    vector<Point> approx;
    approxPolyDP(contour, approx, arcLength(contour, true) * epsilon, true);
    
    // 检查是否有4个顶点
    if (approx.size() != 4) {
        return false;
    }
    
    // 检查是否为凸多边形
    if (!isContourConvex(approx)) {
        return false;
    }
    
    // 计算四条边的长度
    double lengths[4];
    for (int i = 0; i < 4; i++) {
        lengths[i] = norm(approx[i] - approx[(i + 1) % 4]);
    }
    
    // 检查四条边长度是否相近（误差允许20%）
    double avgLength = (lengths[0] + lengths[1] + lengths[2] + lengths[3]) / 4.0;
    for (int i = 0; i < 4; i++) {
        if (abs(lengths[i] - avgLength) > avgLength * 0.2) {
            return false;
        }
    }
    
    return true;
}

// 计算轮廓中心点
Point2f getContourCenter(const vector<Point>& contour) {
    Moments m = moments(contour);
    if (m.m00 != 0) {
        return Point2f(m.m10 / m.m00, m.m01 / m.m00);
    }
    return Point2f(0, 0);
}

// 计算相对于画面中心的弧度值和Y轴偏移
void calculateAngleAndOffset(const Point2f& squareCenter, const Size& imageSize, 
                             float& angle, float& yOffset) {
    // 画面中心点
    Point2f imageCenter(imageSize.width / 2.0f, imageSize.height / 2.0f);
    
    // 计算相对位置
    float dx = squareCenter.x - imageCenter.x;
    float dy = imageCenter.y - squareCenter.y; // Y轴向下为正，所以取反
    
    // 计算弧度值（以竖直轴上半轴为起始，右侧为正，左侧为负）
    // atan2(dx, dy) 计算的是以Y轴正方向为0度，顺时针为正的角度
    angle = atan2(dx, dy);
    
    // Y轴偏移值（正方形中心相对于画面中心的Y轴偏移）
    yOffset = squareCenter.y - imageCenter.y;
}

class SquareDetectionNode : public rclcpp::Node {
public:
    SquareDetectionNode() : Node("square_detection_node"), debug_mode_(true) { // 添加调试模式
        // 创建发布器
        status_pub_ = this->create_publisher<std_msgs::msg::Int32>("square/status", 10);
        angle_pub_ = this->create_publisher<std_msgs::msg::Float32>("square/angle", 10);
        offset_pub_ = this->create_publisher<std_msgs::msg::Float32>("square/offset", 10);
        
        // 使用定时器代替while循环
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&SquareDetectionNode::processFrame, this));
        
        // 打开摄像头
        cap_.open(0);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开摄像头");
            rclcpp::shutdown();
        }
        
        // ===== DEBUG START: 调试窗口和轨迹条 =====
        namedWindow("Square Detection", WINDOW_AUTOSIZE);
        namedWindow("Debug Controls", WINDOW_AUTOSIZE);
        namedWindow("Preprocessed", WINDOW_AUTOSIZE); // 预处理画面窗口
        
        createTrackbar("Canny Low", "Debug Controls", &canny_low_, 255);
        createTrackbar("Canny High", "Debug Controls", &canny_high_, 255);
        createTrackbar("Blur Size", "Debug Controls", &blur_size_, 25);
        createTrackbar("Min Area/100", "Debug Controls", &min_area_, 100);
        createTrackbar("Epsilon x100", "Debug Controls", &epsilon_, 10);
        createTrackbar("Show Edges", "Debug Controls", &show_edges_, 1);
        createTrackbar("Show Preproc", "Debug Controls", &show_preproc_, 1); // 显示预处理画面开关
        // ===== DEBUG END =====
        
        RCLCPP_INFO(this->get_logger(), "正方形检测节点已启动");
    }
    
    ~SquareDetectionNode() {
        cap_.release();
        destroyAllWindows();
    }

private:
    void processFrame() {
        Mat frame, gray, edges;
        
        cap_ >> frame;
        if (frame.empty()) {
            return;
        }
        
        // 转换为灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // ===== DEBUG START: 使用可调参数进行高斯模糊 =====
        int blur = blur_size_ * 2 + 1;
        GaussianBlur(gray, gray, Size(blur, blur), 0);
        
        // 显示预处理后的画面
        if (show_preproc_) {
            imshow("Preprocessed", gray);
        } else {
            destroyWindow("Preprocessed");
        }
        // ===== DEBUG END =====
        // 生产版本使用: GaussianBlur(gray, gray, Size(5, 5), 0);
        
        // ===== DEBUG START: 使用可调参数进行Canny边缘检测 =====
        int low = max(1, canny_low_);
        int high = max(low + 1, canny_high_);
        Canny(gray, edges, low, high);
        // ===== DEBUG END =====
        // 生产版本使用: Canny(gray, edges, 50, 150);
        
        // 查找轮廓
        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        bool found = false;
        float angle = 0.0f, yOffset = 0.0f;
        Point2f center(0, 0);
        // ===== DEBUG START: 调试统计信息 =====
        int contour_count = 0;
        double largest_area = 0;
        // ===== DEBUG END =====
        
        // 查找最大的正方形
        double maxSquareArea = 0;
        vector<Point> maxSquareContour;
        
        // 遍历所有轮廓，查找面积最大的正方形
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            // ===== DEBUG START: 使用可调最小面积参数 =====
            if (area < min_area_ * 100) {
            // ===== DEBUG END =====
            // 生产版本使用: if (area < 1000) {
                continue;
            }
            
            // ===== DEBUG START: 统计轮廓信息 =====
            contour_count++;
            if (area > largest_area) {
                largest_area = area;
            }
            // ===== DEBUG END =====
            
            // ===== DEBUG START: 使用可调epsilon参数 =====
            if (isSquare(contour, epsilon_ / 100.0)) {
            // ===== DEBUG END =====
            // 生产版本使用: if (isSquare(contour)) {
                // 如果找到更大的正方形，更新记录
                if (area > maxSquareArea) {
                    maxSquareArea = area;
                    maxSquareContour = contour;
                    found = true;
                }
            }
        }
        
        // 如果找到最大正方形，进行处理和绘制
        if (found) {
            // ===== DEBUG START: 绘制轮廓 =====
            drawContours(frame, vector<vector<Point>>{maxSquareContour}, -1, Scalar(0, 255, 0), 2);
            // ===== DEBUG END =====
            
            // 计算中心点
            center = getContourCenter(maxSquareContour);
            // ===== DEBUG START: 绘制中心点 =====
            circle(frame, center, 5, Scalar(0, 0, 255), -1);
            // ===== DEBUG END =====
            
            // 计算弧度值和Y轴偏移
            calculateAngleAndOffset(center, frame.size(), angle, yOffset);
            
            // ===== DEBUG START: 在图像上显示检测信息 =====
            char text[100];
            sprintf(text, "Angle: %.2f rad", angle);
            putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            sprintf(text, "Y Offset: %.2f px", yOffset);
            putText(frame, text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            sprintf(text, "Area: %.0f", maxSquareArea);
            putText(frame, text, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            // ===== DEBUG END =====
            
            // ===== DEBUG START: 输出调试信息到终端 =====
            if (debug_mode_) {
                RCLCPP_INFO(this->get_logger(), "状态: 1, 弧度: %.2f, 偏移: %.2f px, 面积: %.0f",
                           angle, yOffset, maxSquareArea);
            }
            // ===== DEBUG END =====
        }
        
        // 发布数据（无论是否找到目标都发布）
        publishSquareData(found, angle, yOffset);
        
        // ===== DEBUG START: 显示检测统计信息 =====
        char stats[100];
        sprintf(stats, "Contours: %d | Max Area: %.0f", contour_count, largest_area);
        putText(frame, stats, Point(10, frame.rows - 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
        sprintf(stats, "Status: %s", found ? "DETECTED" : "SEARCHING");
        putText(frame, stats, Point(10, frame.rows - 10), FONT_HERSHEY_SIMPLEX, 0.6, 
                found ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
        
        if (!found && debug_mode_) {
            RCLCPP_INFO(this->get_logger(), "未检测到正方形 (轮廓数: %d)", contour_count);
        }
        // ===== DEBUG END =====
        
        // ===== DEBUG START: 绘制画面中心点和十字线 =====
        Point imageCenter(frame.cols / 2, frame.rows / 2);
        circle(frame, imageCenter, 5, Scalar(255, 0, 0), -1);
        line(frame, Point(imageCenter.x, 0), Point(imageCenter.x, frame.rows), Scalar(255, 0, 0), 1);
        line(frame, Point(0, imageCenter.y), Point(frame.cols, imageCenter.y), Scalar(255, 0, 0), 1);
        // ===== DEBUG END =====
        
        // ===== DEBUG START: 显示检测结果窗口 =====
        imshow("Square Detection", frame);
        
        // 显示边缘检测结果（可选）
        if (show_edges_) {
            imshow("Edges", edges);
        } else {
            destroyWindow("Edges");
        }
        
        waitKey(1);
        // ===== DEBUG END =====
    }
    
    void publishSquareData(bool hasTarget, float angle, float yOffset) {
        // 发布状态：1表示有目标，0表示没有目标
        auto status_msg = std_msgs::msg::Int32();
        status_msg.data = hasTarget ? 1 : 0;
        status_pub_->publish(status_msg);
        
        // 发布角度（保留两位小数）
        auto angle_msg = std_msgs::msg::Float32();
        angle_msg.data = std::round(angle * 100.0f) / 100.0f;
        angle_pub_->publish(angle_msg);
        
        // 发布偏移值（像素单位，保留两位小数）
        auto offset_msg = std_msgs::msg::Float32();
        offset_msg.data = std::round(yOffset * 100.0f) / 100.0f;
        offset_pub_->publish(offset_msg);
    }
    
    VideoCapture cap_;
    bool debug_mode_; // ===== DEBUG: 调试模式标志 =====
    
    // ===== DEBUG START: 可调试参数 =====
    int canny_low_ = 50;
    int canny_high_ = 150;
    int blur_size_ = 2;  // 实际大小 = blur_size_ * 2 + 1
    int min_area_ = 10;  // 实际面积 = min_area_ * 100
    int epsilon_ = 2;    // 实际值 = epsilon_ / 100.0
    int show_edges_ = 0;
    int show_preproc_ = 1; // 显示预处理画面开关
    // ===== DEBUG END =====
    
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr status_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr angle_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SquareDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
