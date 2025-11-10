#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace cv;
using namespace std::chrono_literals;

// 颜色模式
enum class ColorMode { Red, Blue, Both };

// 参数结构体
struct TrackingParams {
    int min_area = 2000;
    int max_area = 50000;
    double send_rate = 50.0;
    ColorMode color_mode = ColorMode::Both;
    bool publish_image = true;
    bool publish_mask = false;
    bool show_window = true;
    int smoothing_window = 5;
    double contour_epsilon = 0.015;
};

// 轮廓平滑函数
static void smoothContour(std::vector<cv::Point>& contour, double epsilon_ratio = 0.01) {
    if (contour.size() < 3) return;
    
    std::vector<cv::Point> approximated;
    double arc_length = cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approximated, arc_length * epsilon_ratio, true);
    
    contour = approximated;
}

// 创建颜色掩码
static void makeMaskHSV(const cv::Mat& hsv, ColorMode mode, cv::Mat& maskOut) {
    cv::Mat maskRed, maskBlue;

    if (mode == ColorMode::Red || mode == ColorMode::Both) {
        cv::Mat lower, upper;
        cv::inRange(hsv, cv::Scalar(0,   120, 70), cv::Scalar(10,  255, 255), lower);
        cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), upper);
        cv::bitwise_or(lower, upper, maskRed);
    }
    if (mode == ColorMode::Blue || mode == ColorMode::Both) {
        cv::inRange(hsv, cv::Scalar(100, 120, 60), cv::Scalar(140, 255, 255), maskBlue);
    }

    if (mode == ColorMode::Red) maskOut = maskRed;
    else if (mode == ColorMode::Blue) maskOut = maskBlue;
    else cv::bitwise_or(maskRed, maskBlue, maskOut);

    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    
    cv::morphologyEx(maskOut, maskOut, cv::MORPH_OPEN, kernel_open, cv::Point(-1,-1), 2);
    cv::morphologyEx(maskOut, maskOut, cv::MORPH_CLOSE, kernel_close, cv::Point(-1,-1), 2);
    cv::threshold(maskOut, maskOut, 127, 255, cv::THRESH_BINARY);
}

// 在 mask 中查找最大轮廓及其中心
static bool findLargestCenter(const cv::Mat& mask, cv::Point2f& center, 
                             std::vector<cv::Point>& bestContour, 
                             double minArea = 800.0) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    double bestArea = 0.0; 
    bestContour.clear();
    
    for (const auto& c : contours) {
        double a = cv::contourArea(c);
        if (a > minArea && a > bestArea) { 
            bestArea = a; 
            bestContour = c; 
        }
    }
    
    if (bestContour.empty()) return false;
    
    smoothContour(bestContour, 0.015);
    
    std::vector<cv::Point> hull;
    cv::convexHull(bestContour, hull);
    if (hull.size() >= 3) {
        bestContour = hull;
    }
    
    cv::Moments m = cv::moments(bestContour);
    if (std::abs(m.m00) < 1e-5) return false;
    
    center = cv::Point2f(static_cast<float>(m.m10 / m.m00), 
                         static_cast<float>(m.m01 / m.m00));
    return true;
}

// 中心点平滑滤波器
class CenterSmoother {
private:
    std::vector<cv::Point2f> centers_;
    size_t max_size_;
    
public:
    CenterSmoother(size_t max_size = 5) : max_size_(max_size) {}
    
    cv::Point2f smooth(const cv::Point2f& new_center, bool valid) {
        if (!valid) {
            centers_.clear();
            return cv::Point2f(0, 0);
        }
        
        centers_.push_back(new_center);
        if (centers_.size() > max_size_) {
            centers_.erase(centers_.begin());
        }
        
        float x = 0, y = 0;
        for (const auto& center : centers_) {
            x += center.x;
            y += center.y;
        }
        return cv::Point2f(x / centers_.size(), y / centers_.size());
    }
};

// 计算目标点相对于画面中心竖直坐标轴上半轴的偏转弧度值
static double calculateDeflectionRadians(const cv::Point2f& target, const cv::Point2f& center, int imageHeight) {
    float dx = target.x - center.x;
    float dy = center.y - target.y;
    
    if (std::abs(dx) < 1e-5 && std::abs(dy) < 1e-5) {
        return 0.0;
    }
    
    if (std::abs(dx) < 1e-5) {
        return (dy > 0) ? 0.0 : CV_PI;
    }
    
    if (std::abs(dy) < 1e-5) {
        return (dx > 0) ? CV_PI/2 : -CV_PI/2;
    }
    
    double angle = std::atan2(dx, dy);
    
    if (angle < -CV_PI) angle += 2 * CV_PI;
    if (angle > CV_PI) angle -= 2 * CV_PI;
    
    return angle;
}

// 将弧度值转换为有意义的偏转角度描述
static std::string getAngleDescription(double radians) {
    double degrees = radians * 180.0 / CV_PI;
    
    if (std::abs(degrees) < 5.0) return "zs";
    else if (degrees > 5.0 && degrees < 85.0) return "ys";
    else if (std::abs(degrees - 90.0) < 5.0) return "y";
    else if (degrees > 85.0 && degrees < 95.0) return "y";
    else if (degrees > 95.0 && degrees < 175.0) return "yx";
    else if (std::abs(degrees - 180.0) < 5.0 || std::abs(degrees + 180.0) < 5.0) return "x";
    else if (degrees < -95.0 && degrees > -175.0) return "zx";
    else if (std::abs(degrees + 90.0) < 5.0) return "z";
    else if (degrees < -5.0 && degrees > -85.0) return "zs";
    else return "z";
    
    return "?";
}

// 主节点类
class ColorTrackingNode : public rclcpp::Node {
public:
    ColorTrackingNode() : Node("color_tracking_node"), 
                         window_initialized_(false),
                         last_frame_time_(std::chrono::steady_clock::now()) {
        // 参数声明
        this->declare_parameter<int>("camera_index", 0);
        this->declare_parameter<int>("min_area", 2000);
        this->declare_parameter<int>("max_area", 50000);
        this->declare_parameter<double>("publish_rate", 50.0);
        this->declare_parameter<std::string>("color_mode", "both");
        this->declare_parameter<bool>("publish_image", true);
        this->declare_parameter<bool>("publish_mask", false);
        this->declare_parameter<bool>("show_window", true);
        this->declare_parameter<int>("smoothing_window", 5);
        this->declare_parameter<double>("contour_epsilon", 0.015);
        this->declare_parameter<int>("image_width", 640);
        this->declare_parameter<int>("image_height", 480);
        this->declare_parameter<int>("fps", 30);

        // 读取参数
        loadParameters();

        // 图像传输 - 修复：延迟初始化，不在构造函数中使用shared_from_this()
        // image_transport_ 将在 getImageTransport() 中延迟初始化

        // 发布器
        result_publisher_ = this->create_publisher<std_msgs::msg::String>(
            "color_tracking/result", 10);
        
        // 注意：image_publisher_ 和 mask_publisher_ 也将延迟初始化

        // 定时器
        auto timer_rate = std::chrono::duration<double>(1.0 / params_.send_rate);
        timer_ = this->create_wall_timer(
            timer_rate, std::bind(&ColorTrackingNode::processingLoop, this));

        // 初始化相机
        initializeCamera();

        // 中心点平滑器
        center_smoother_ = std::make_unique<CenterSmoother>(params_.smoothing_window);

        RCLCPP_INFO(this->get_logger(), "Color Tracking Node 初始化完成");
        RCLCPP_INFO(this->get_logger(), "相机索引: %d", camera_index_);
        RCLCPP_INFO(this->get_logger(), "发布频率: %.1f Hz", params_.send_rate);
        RCLCPP_INFO(this->get_logger(), "显示窗口: %s", params_.show_window ? "是" : "否");
        RCLCPP_INFO(this->get_logger(), "颜色模式: %s", 
                   params_.color_mode == ColorMode::Red ? "RED" : 
                   params_.color_mode == ColorMode::Blue ? "BLUE" : "BOTH");
    }

    ~ColorTrackingNode() {
        if (cap_.isOpened()) {
            cap_.release();
            RCLCPP_INFO(this->get_logger(), "相机资源已释放");
        }
        if (window_initialized_) {
            cv::destroyAllWindows();
            RCLCPP_INFO(this->get_logger(), "OpenCV窗口已关闭");
        }
    }

private:
    void loadParameters() {
        camera_index_ = this->get_parameter("camera_index").as_int();
        
        params_.min_area = this->get_parameter("min_area").as_int();
        params_.max_area = this->get_parameter("max_area").as_int();
        params_.send_rate = this->get_parameter("publish_rate").as_double();
        params_.publish_image = this->get_parameter("publish_image").as_bool();
        params_.publish_mask = this->get_parameter("publish_mask").as_bool();
        params_.show_window = this->get_parameter("show_window").as_bool();
        params_.smoothing_window = this->get_parameter("smoothing_window").as_int();
        params_.contour_epsilon = this->get_parameter("contour_epsilon").as_double();
        
        std::string color_mode_str = this->get_parameter("color_mode").as_string();
        if (color_mode_str == "red") params_.color_mode = ColorMode::Red;
        else if (color_mode_str == "blue") params_.color_mode = ColorMode::Blue;
        else params_.color_mode = ColorMode::Both;

        image_width_ = this->get_parameter("image_width").as_int();
        image_height_ = this->get_parameter("image_height").as_int();
        fps_ = this->get_parameter("fps").as_int();
    }

    void initializeCamera() {
        cap_.open(camera_index_);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开相机: index=%d", camera_index_);
            return;
        }

        cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
        cap_.set(cv::CAP_PROP_FPS, fps_);

        RCLCPP_INFO(this->get_logger(), "相机初始化成功: %dx%d @ %d FPS", 
                   image_width_, image_height_, fps_);
    }

    void processingLoop() {
        if (!cap_.isOpened()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                               "相机未就绪");
            return;
        }

        cv::Mat frame, hsv, mask;
        if (!cap_.read(frame) || frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "读取帧失败");
            return;
        }

        // 图像处理
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        makeMaskHSV(hsv, params_.color_mode, mask);

        // 目标检测
        cv::Point2f raw_center, smoothed_center;
        std::vector<cv::Point> contour;
        bool found = findLargestCenter(mask, raw_center, contour, params_.min_area);
        
        smoothed_center = center_smoother_->smooth(raw_center, found);

        int w = frame.cols, h = frame.rows;
        cv::Point2f image_center(w * 0.5f, h * 0.5f);
        
        double deflection_rad = 0.0;
        int dy = 0;
        bool valid = false;
        
        if (found) {
            deflection_rad = calculateDeflectionRadians(smoothed_center, image_center, h);
            dy = static_cast<int>(std::round(smoothed_center.y - image_center.y));
            valid = true;
        }

        // 发布结果
        publishResult(valid, deflection_rad, dy);
        
        // 可视化
        visualizeResults(frame, valid, contour, smoothed_center, deflection_rad, dy);
        publishImages(frame, mask);

        // 显示窗口
        if (params_.show_window) {
            showCameraWindow(frame, valid, deflection_rad, dy);
        }
    }

    void showCameraWindow(cv::Mat& frame, bool valid, double deflection_rad, int dy) {
        if (!window_initialized_) {
            cv::namedWindow("Color Tracking - Press 'q' to quit", cv::WINDOW_AUTOSIZE);
            window_initialized_ = true;
        }
        
        // 计算FPS
        auto now = std::chrono::steady_clock::now();
        double fps = 1.0 / std::chrono::duration<double>(now - last_frame_time_).count();
        last_frame_time_ = now;
        
        std::ostringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(frame, fps_text.str(), cv::Point(10, frame.rows - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        
        cv::imshow("Color Tracking - Press 'q' to quit", frame);
        
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            RCLCPP_INFO(this->get_logger(), "用户请求退出");
            rclcpp::shutdown();
        }
    }

    void publishResult(bool valid, double deflection_rad, int dy) {
        auto string_msg = std_msgs::msg::String();
        std::ostringstream oss;

        oss << std::fixed << std::setprecision(2);
        
        float status = valid ? 1.00f : 0.00f;
        float rad = valid ? static_cast<float>(deflection_rad) : 0.00f;
        float y_offset = valid ? static_cast<float>(dy) : 0.00f;
        
        oss << status << "," << rad << "," << y_offset;

        string_msg.data = oss.str();
        result_publisher_->publish(string_msg);

        if (valid) {
            RCLCPP_DEBUG(this->get_logger(), "目标检测: rad=%.2f, dy=%d", 
                        deflection_rad, dy);
        }
    }

    void visualizeResults(cv::Mat& frame, bool valid,
                         const std::vector<cv::Point>& contour,
                         const cv::Point2f& center,
                         double deflection_rad,
                         int dy) {
        int w = frame.cols, h = frame.rows;
        cv::Point2f image_center(w * 0.5f, h * 0.5f);
        
        if (valid) {
            if (!contour.empty()) {
                std::vector<std::vector<cv::Point>> contours_to_draw = {contour};
                cv::drawContours(frame, contours_to_draw, -1, cv::Scalar(0, 100, 0), -1);
                cv::drawContours(frame, contours_to_draw, -1, cv::Scalar(0, 255, 0), 3);
            }
            
            cv::circle(frame, center, 6, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(frame, center, 3, cv::Scalar(255, 255, 255), 2);
            
            cv::arrowedLine(frame, image_center, center, cv::Scalar(255, 255, 255), 2);
            
            int line_length = 150;
            cv::Point deflection_end(
                image_center.x + static_cast<int>(line_length * std::sin(deflection_rad)),
                image_center.y - static_cast<int>(line_length * std::cos(deflection_rad))
            );
            cv::arrowedLine(frame, image_center, deflection_end, cv::Scalar(0, 255, 255), 3);
            
            int arc_radius = 80;
            cv::ellipse(frame, image_center, cv::Size(arc_radius, arc_radius), 0, 0, 
                       deflection_rad * 180.0 / CV_PI, cv::Scalar(255, 100, 0), 2);
        }
        
        cv::line(frame, cv::Point(image_center.x-20, image_center.y), 
                cv::Point(image_center.x+20, image_center.y), cv::Scalar(255,255,255), 2);
        cv::line(frame, cv::Point(image_center.x, image_center.y-20), 
                cv::Point(image_center.x, image_center.y+20), cv::Scalar(255,255,255), 2);
        
        cv::line(frame, image_center, cv::Point(image_center.x, 0), cv::Scalar(100, 100, 255), 2);
        
        cv::line(frame, cv::Point(0, image_center.y), cv::Point(w, image_center.y), 
                cv::Scalar(100, 100, 100), 1);
        cv::line(frame, cv::Point(image_center.x, 0), cv::Point(image_center.x, h), 
                cv::Scalar(100, 100, 100), 1);
        
        std::ostringstream info;
        double deflection_deg = deflection_rad * 180.0 / CV_PI;
        info << (params_.color_mode==ColorMode::Red?"RED":
                params_.color_mode==ColorMode::Blue?"BLUE":"BOTH")
             << " | valid=" << (valid?1:0) 
             << " rad=" << std::fixed << std::setprecision(4) << deflection_rad
             << " (" << std::setprecision(1) << deflection_deg << "du)";
        cv::putText(frame, info.str(), cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        
        if (valid) {
            std::ostringstream dir_info;
            dir_info << "fangxiang: " << getAngleDescription(deflection_rad) << " | Y: " << dy << "px";
            cv::putText(frame, dir_info.str(), cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 2);
        }

        std::ostringstream quadrant_info;
        if (valid) {
            double deg = deflection_deg;
            if (deg >= -180 && deg < -90) quadrant_info << "zx";
            else if (deg >= -90 && deg < 0) quadrant_info << "zs";
            else if (deg >= 0 && deg < 90) quadrant_info << "ys";
            else quadrant_info << "yx";
        } else {
            quadrant_info << "none";
        }
        cv::putText(frame, quadrant_info.str(), cv::Point(10, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);
    }

    void publishImages(const cv::Mat& frame, const cv::Mat& mask) {
        if (params_.publish_image) {
            if (!image_publisher_) {
                image_publisher_ = getImageTransport().advertise("color_tracking/image", 1);
            }
            auto image_msg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            image_msg->header.stamp = this->now();
            image_msg->header.frame_id = "camera";
            image_publisher_.publish(image_msg);
        }
        
        if (params_.publish_mask) {
            if (!mask_publisher_) {
                mask_publisher_ = getImageTransport().advertise("color_tracking/mask", 1);
            }
            cv::Mat mask_bgr;
            cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
            auto mask_msg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgr8", mask_bgr).toImageMsg();
            mask_msg->header.stamp = this->now();
            mask_msg->header.frame_id = "camera";
            mask_publisher_.publish(mask_msg);
        }
    }

    // 延迟初始化 image_transport
    image_transport::ImageTransport& getImageTransport() {
        if (!image_transport_) {
            image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
        }
        return *image_transport_;
    }

    // 成员变量
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Publisher image_publisher_;
    image_transport::Publisher mask_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    cv::VideoCapture cap_;
    std::unique_ptr<CenterSmoother> center_smoother_;
    bool window_initialized_;
    std::chrono::steady_clock::time_point last_frame_time_;
    
    TrackingParams params_;
    int camera_index_;
    int image_width_;
    int image_height_;
    int fps_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ColorTrackingNode>();
    rclcpp::spin(node);
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}
