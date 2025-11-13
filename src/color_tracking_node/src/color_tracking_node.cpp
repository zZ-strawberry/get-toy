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

// 形状类型
enum class ShapeType { 
    Triangle,   // 三角形
    Rectangle,  // 矩形
    Circle,     // 圆形
    Unknown     // 未知形状（用于最规整轮廓）
};

// 目标信息结构体
struct TargetInfo {
    cv::Point2f center;
    std::vector<cv::Point> contour;
    ShapeType shape;
    ColorMode color;
    double regularity_score;  // 规整度分数 (0-1)
    double area;
    bool is_valid;
    
    TargetInfo() : shape(ShapeType::Unknown), color(ColorMode::Both), 
                  regularity_score(0.0), area(0.0), is_valid(false) {}
};

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
    
    // 形状识别参数
    bool enable_shape_detection = true;
    bool track_triangle = true;
    bool track_rectangle = true;
    bool track_circle = true;
    double min_regularity = 0.5;  // 最小规整度阈值
    double circularity_threshold = 0.7;  // 圆形度阈值
    double rectangularity_threshold = 0.75;  // 矩形度阈值
};

// 轮廓平滑函数
static void smoothContour(std::vector<cv::Point>& contour, double epsilon_ratio = 0.01) {
    if (contour.size() < 3) return;
    
    std::vector<cv::Point> approximated;
    double arc_length = cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approximated, arc_length * epsilon_ratio, true);
    
    contour = approximated;
}

// 计算轮廓的规整度分数 (0-1)
static double calculateRegularityScore(const std::vector<cv::Point>& contour) {
    if (contour.size() < 3) return 0.0;
    
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter < 1e-5) return 0.0;
    
    // 使用圆形度作为规整度的基础指标
    // 圆形度 = 4π * 面积 / 周长²，完美的圆为1
    double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    
    // 凸包面积比
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hull_area = cv::contourArea(hull);
    double convexity = (hull_area > 0) ? (area / hull_area) : 0.0;
    
    // 综合评分
    return std::min(1.0, (circularity * 0.6 + convexity * 0.4));
}

// 识别形状类型
static ShapeType recognizeShape(const std::vector<cv::Point>& contour, 
                               double circularity_threshold = 0.7,
                               double rectangularity_threshold = 0.75) {
    if (contour.size() < 3) return ShapeType::Unknown;
    
    // 多边形近似
    std::vector<cv::Point> approx;
    double epsilon = 0.04 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);
    
    int vertices = approx.size();
    
    // 计算圆形度
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    double circularity = (perimeter > 0) ? (4.0 * CV_PI * area / (perimeter * perimeter)) : 0.0;
    
    // 圆形判断
    if (circularity >= circularity_threshold && vertices > 6) {
        return ShapeType::Circle;
    }
    
    // 三角形判断
    if (vertices == 3) {
        return ShapeType::Triangle;
    }
    
    // 矩形判断
    if (vertices == 4) {
        // 计算矩形度
        cv::RotatedRect rect = cv::minAreaRect(contour);
        double rect_area = rect.size.width * rect.size.height;
        double rectangularity = (rect_area > 0) ? (area / rect_area) : 0.0;
        
        if (rectangularity >= rectangularity_threshold) {
            return ShapeType::Rectangle;
        }
    }
    
    return ShapeType::Unknown;
}

// 获取形状名称
static std::string getShapeName(ShapeType shape) {
    switch (shape) {
        case ShapeType::Triangle: return "Triangle";
        case ShapeType::Rectangle: return "Rectangle";
        case ShapeType::Circle: return "Circle";
        case ShapeType::Unknown: return "Unknown";
        default: return "Unknown";
    }
}

// 判断颜色（通过HSV掩码位置）
static ColorMode detectColor(const cv::Point2f& center, const cv::Mat& maskRed, const cv::Mat& maskBlue) {
    int x = static_cast<int>(center.x);
    int y = static_cast<int>(center.y);
    
    if (x < 0 || x >= maskRed.cols || y < 0 || y >= maskRed.rows) {
        return ColorMode::Both;
    }
    
    bool is_red = maskRed.at<uchar>(y, x) > 127;
    bool is_blue = maskBlue.at<uchar>(y, x) > 127;
    
    if (is_red && !is_blue) return ColorMode::Red;
    if (is_blue && !is_red) return ColorMode::Blue;
    return ColorMode::Both;
}

// 计算图像平均亮度用于自适应调整
static double calculateAverageBrightness(const cv::Mat& hsv) {
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Scalar mean_val = cv::mean(channels[2]); // V通道
    return mean_val[0];
}

// 创建颜色掩码（分别返回红色和蓝色掩码）- 改进版本，增强蓝黑区分能力
static void makeMaskHSV(const cv::Mat& hsv, ColorMode mode, cv::Mat& maskOut, 
                       cv::Mat& maskRed, cv::Mat& maskBlue) {
    maskRed = cv::Mat::zeros(hsv.size(), CV_8UC1);
    maskBlue = cv::Mat::zeros(hsv.size(), CV_8UC1);

    // 计算环境亮度进行自适应调整
    double avg_brightness = calculateAverageBrightness(hsv);
    
    // 根据环境亮度动态调整阈值
    int red_min_v = 70;
    int blue_min_s = 150;  // 提高饱和度下限，排除黑色
    int blue_min_v = 80;   // 提高亮度下限，排除黑色
    int blue_max_v = 255;
    
    // 光线自适应：暗环境下降低亮度要求，但保持高饱和度要求
    if (avg_brightness < 80) {
        // 暗环境
        red_min_v = 50;
        blue_min_v = 60;
        blue_min_s = 140;
        blue_max_v = 220;
    } else if (avg_brightness > 180) {
        // 亮环境
        red_min_v = 90;
        blue_min_v = 100;
        blue_min_s = 130;
        blue_max_v = 255;
    }

    if (mode == ColorMode::Red || mode == ColorMode::Both) {
        cv::Mat lower, upper;
        cv::inRange(hsv, cv::Scalar(0,   120, red_min_v), cv::Scalar(10,  255, 255), lower);
        cv::inRange(hsv, cv::Scalar(170, 120, red_min_v), cv::Scalar(180, 255, 255), upper);
        cv::bitwise_or(lower, upper, maskRed);
    }
    
    if (mode == ColorMode::Blue || mode == ColorMode::Both) {
        // 蓝色检测：严格的饱和度和亮度范围，排除黑色
        // 黑色特征：低饱和度(S<50)、低亮度(V<50)
        // 蓝色特征：高饱和度(S>130)、中高亮度(V>60)
        cv::Mat blue_mask_primary, blue_mask_refined;
        
        // 第一步：基于色调和高饱和度筛选
        cv::inRange(hsv, cv::Scalar(100, blue_min_s, blue_min_v), 
                   cv::Scalar(130, 255, blue_max_v), blue_mask_primary);
        
        // 第二步：排除黑色区域（低饱和度+低亮度）
        cv::Mat black_mask;
        cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 50, 50), black_mask);
        
        // 从蓝色掩码中减去黑色掩码
        cv::bitwise_and(blue_mask_primary, cv::Scalar(255) - black_mask, maskBlue);
        
        // 第三步：额外的形态学操作去除噪声
        cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(maskBlue, maskBlue, kernel_erode, cv::Point(-1,-1), 1);
    }

    if (mode == ColorMode::Red) maskOut = maskRed.clone();
    else if (mode == ColorMode::Blue) maskOut = maskBlue.clone();
    else cv::bitwise_or(maskRed, maskBlue, maskOut);

    // 形态学操作优化
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    
    cv::morphologyEx(maskOut, maskOut, cv::MORPH_OPEN, kernel_open, cv::Point(-1,-1), 2);
    cv::morphologyEx(maskOut, maskOut, cv::MORPH_CLOSE, kernel_close, cv::Point(-1,-1), 2);
    cv::threshold(maskOut, maskOut, 127, 255, cv::THRESH_BINARY);
    
    // 对单独的红蓝掩码也做处理
    if (!maskRed.empty() && cv::countNonZero(maskRed) > 0) {
        cv::morphologyEx(maskRed, maskRed, cv::MORPH_OPEN, kernel_open, cv::Point(-1,-1), 2);
        cv::morphologyEx(maskRed, maskRed, cv::MORPH_CLOSE, kernel_close, cv::Point(-1,-1), 2);
    }
    if (!maskBlue.empty() && cv::countNonZero(maskBlue) > 0) {
        cv::morphologyEx(maskBlue, maskBlue, cv::MORPH_OPEN, kernel_open, cv::Point(-1,-1), 2);
        cv::morphologyEx(maskBlue, maskBlue, cv::MORPH_CLOSE, kernel_close, cv::Point(-1,-1), 2);
    }
}

// 查找最佳目标（优先匹配指定形状，否则选择最规整的轮廓）
static bool findBestTarget(const cv::Mat& mask, 
                          const cv::Mat& maskRed, 
                          const cv::Mat& maskBlue,
                          const TrackingParams& params,
                          TargetInfo& target) {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat mask_copy = mask.clone();
    cv::findContours(mask_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        target.is_valid = false;
        return false;
    }
    
    // 候选目标列表
    std::vector<TargetInfo> candidates;
    
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        
        // 面积过滤
        if (area < params.min_area || area > params.max_area) continue;
        
        TargetInfo candidate;
        candidate.contour = c;
        candidate.area = area;
        
        // 平滑轮廓
        smoothContour(candidate.contour, params.contour_epsilon);
        
        // 计算凸包
        std::vector<cv::Point> hull;
        cv::convexHull(candidate.contour, hull);
        if (hull.size() >= 3) {
            candidate.contour = hull;
        }
        
        // 计算中心
        cv::Moments m = cv::moments(candidate.contour);
        if (std::abs(m.m00) < 1e-5) continue;
        candidate.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), 
                                       static_cast<float>(m.m01 / m.m00));
        
        // 识别形状
        if (params.enable_shape_detection) {
            candidate.shape = recognizeShape(candidate.contour, 
                                            params.circularity_threshold,
                                            params.rectangularity_threshold);
        } else {
            candidate.shape = ShapeType::Unknown;
        }
        
        // 计算规整度
        candidate.regularity_score = calculateRegularityScore(candidate.contour);
        
        // 检测颜色
        candidate.color = detectColor(candidate.center, maskRed, maskBlue);
        
        candidate.is_valid = true;
        candidates.push_back(candidate);
    }
    
    if (candidates.empty()) {
        target.is_valid = false;
        return false;
    }
    
    // 选择最佳目标
    TargetInfo* best = nullptr;
    double best_score = -1.0;
    
    for (auto& candidate : candidates) {
        double score = 0.0;
        
        // 优先级1: 匹配指定形状
        bool shape_match = false;
        if (params.enable_shape_detection) {
            if ((candidate.shape == ShapeType::Triangle && params.track_triangle) ||
                (candidate.shape == ShapeType::Rectangle && params.track_rectangle) ||
                (candidate.shape == ShapeType::Circle && params.track_circle)) {
                shape_match = true;
                score += 100.0;  // 形状匹配加高分
            }
        }
        
        // 优先级2: 如果没有形状匹配，使用规整度
        if (!shape_match) {
            score += candidate.regularity_score * 50.0;
        }
        
        // 额外分数：面积（大目标优先）
        score += (candidate.area / params.max_area) * 10.0;
        
        // 额外分数：规整度
        score += candidate.regularity_score * 5.0;
        
        if (score > best_score) {
            best_score = score;
            best = &candidate;
        }
    }
    
    if (best && best->is_valid) {
        target = *best;
        return true;
    }
    
    target.is_valid = false;
    return false;
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
    (void)imageHeight;  // 参数保留供未来使用
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
        
        // 形状识别参数
        this->declare_parameter<bool>("enable_shape_detection", true);
        this->declare_parameter<bool>("track_triangle", true);
        this->declare_parameter<bool>("track_rectangle", true);
        this->declare_parameter<bool>("track_circle", true);
        this->declare_parameter<double>("min_regularity", 0.5);
        this->declare_parameter<double>("circularity_threshold", 0.7);
        this->declare_parameter<double>("rectangularity_threshold", 0.75);

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
        RCLCPP_INFO(this->get_logger(), "最小面积阈值: %d 像素", params_.min_area);
        RCLCPP_INFO(this->get_logger(), "最大面积阈值: %d 像素", params_.max_area);
        
        if (params_.enable_shape_detection) {
            RCLCPP_INFO(this->get_logger(), "形状识别: 启用 (三角形:%s 矩形:%s 圆形:%s)", 
                       params_.track_triangle ? "√" : "×",
                       params_.track_rectangle ? "√" : "×",
                       params_.track_circle ? "√" : "×");
            RCLCPP_INFO(this->get_logger(), "最小规整度: %.2f", params_.min_regularity);
        } else {
            RCLCPP_INFO(this->get_logger(), "形状识别: 禁用 (追踪最规整轮廓)");
        }
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
        
        // 形状识别参数
        params_.enable_shape_detection = this->get_parameter("enable_shape_detection").as_bool();
        params_.track_triangle = this->get_parameter("track_triangle").as_bool();
        params_.track_rectangle = this->get_parameter("track_rectangle").as_bool();
        params_.track_circle = this->get_parameter("track_circle").as_bool();
        params_.min_regularity = this->get_parameter("min_regularity").as_double();
        params_.circularity_threshold = this->get_parameter("circularity_threshold").as_double();
        params_.rectangularity_threshold = this->get_parameter("rectangularity_threshold").as_double();
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

        cv::Mat frame, hsv, mask, maskRed, maskBlue;
        if (!cap_.read(frame) || frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "读取帧失败");
            return;
        }

        // 图像预处理：光线归一化和颜色增强
        cv::Mat frame_enhanced = frame.clone();
        
        // 1. 使用CLAHE (限制对比度自适应直方图均衡化) 改善光照不均
        cv::Mat lab;
        cv::cvtColor(frame_enhanced, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, frame_enhanced, cv::COLOR_Lab2BGR);
        
        // 2. 轻微锐化以增强边缘
        cv::Mat kernel = (cv::Mat_<float>(3,3) << 
            0, -0.5, 0,
            -0.5, 3, -0.5,
            0, -0.5, 0);
        cv::Mat sharpened;
        cv::filter2D(frame_enhanced, sharpened, -1, kernel);
        cv::addWeighted(frame_enhanced, 0.7, sharpened, 0.3, 0, frame_enhanced);
        
        // 3. 转换到HSV色彩空间
        cv::cvtColor(frame_enhanced, hsv, cv::COLOR_BGR2HSV);
        
        // 创建颜色掩码
        makeMaskHSV(hsv, params_.color_mode, mask, maskRed, maskBlue);

        // 目标检测 - 使用新的形状识别系统
        TargetInfo target;
        bool found = findBestTarget(mask, maskRed, maskBlue, params_, target);
        
        // 平滑中心点
        cv::Point2f smoothed_center = center_smoother_->smooth(target.center, found);
        if (found) {
            target.center = smoothed_center;
        }

        int w = frame.cols, h = frame.rows;
        cv::Point2f image_center(w * 0.5f, h * 0.5f);
        
        double deflection_rad = 0.0;
        int dy = 0;
        bool valid = found && target.is_valid;
        
        if (valid) {
            deflection_rad = calculateDeflectionRadians(target.center, image_center, h);
            dy = static_cast<int>(std::round(target.center.y - image_center.y));
        }

        // 发布结果
        publishResult(valid, deflection_rad, dy);
        
        // 获取环境亮度用于显示
        double avg_brightness = calculateAverageBrightness(hsv);
        
        // 可视化
        visualizeResults(frame, target, deflection_rad, dy, avg_brightness);
        publishImages(frame, mask);

        // 显示窗口
        if (params_.show_window) {
            showCameraWindow(frame, valid, deflection_rad, dy);
        }
    }

    void showCameraWindow(cv::Mat& frame, bool valid, double deflection_rad, int dy) {
        (void)valid;  // 参数已在visualizeResults中使用，此处保留接口一致性
        (void)deflection_rad;
        (void)dy;
        
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

    void visualizeResults(cv::Mat& frame, const TargetInfo& target,
                         double deflection_rad, int dy, double avg_brightness = -1.0) {
        int w = frame.cols, h = frame.rows;
        cv::Point2f image_center(w * 0.5f, h * 0.5f);
        bool valid = target.is_valid;
        
        if (valid) {
            if (!target.contour.empty()) {
                std::vector<std::vector<cv::Point>> contours_to_draw = {target.contour};
                
                // 根据颜色选择不同的显示颜色
                cv::Scalar fill_color, outline_color;
                if (target.color == ColorMode::Red) {
                    fill_color = cv::Scalar(0, 0, 150);  // 深红色填充
                    outline_color = cv::Scalar(0, 0, 255);  // 红色轮廓
                } else if (target.color == ColorMode::Blue) {
                    fill_color = cv::Scalar(150, 0, 0);  // 深蓝色填充
                    outline_color = cv::Scalar(255, 0, 0);  // 蓝色轮廓
                } else {
                    fill_color = cv::Scalar(0, 100, 0);  // 深绿色填充
                    outline_color = cv::Scalar(0, 255, 0);  // 绿色轮廓
                }
                
                cv::drawContours(frame, contours_to_draw, -1, fill_color, -1);
                cv::drawContours(frame, contours_to_draw, -1, outline_color, 3);
            }
            
            cv::circle(frame, target.center, 6, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(frame, target.center, 3, cv::Scalar(255, 255, 255), 2);
            
            cv::arrowedLine(frame, image_center, target.center, cv::Scalar(255, 255, 255), 2);
            
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
        
        // 绘制中心十字线
        cv::line(frame, cv::Point(image_center.x-20, image_center.y), 
                cv::Point(image_center.x+20, image_center.y), cv::Scalar(255,255,255), 2);
        cv::line(frame, cv::Point(image_center.x, image_center.y-20), 
                cv::Point(image_center.x, image_center.y+20), cv::Scalar(255,255,255), 2);
        
        cv::line(frame, image_center, cv::Point(image_center.x, 0), cv::Scalar(100, 100, 255), 2);
        
        cv::line(frame, cv::Point(0, image_center.y), cv::Point(w, image_center.y), 
                cv::Scalar(100, 100, 100), 1);
        cv::line(frame, cv::Point(image_center.x, 0), cv::Point(image_center.x, h), 
                cv::Scalar(100, 100, 100), 1);
        
        // 顶部信息：模式和状态
        std::ostringstream info;
        double deflection_deg = deflection_rad * 180.0 / CV_PI;
        info << (params_.color_mode==ColorMode::Red?"RED":
                params_.color_mode==ColorMode::Blue?"BLUE":"BOTH")
             << " | valid=" << (valid?1:0) 
             << " rad=" << std::fixed << std::setprecision(4) << deflection_rad
             << " (" << std::setprecision(1) << deflection_deg << "du)";
        if (avg_brightness >= 0) {
            info << " | Brightness=" << std::setprecision(0) << avg_brightness;
        }
        cv::putText(frame, info.str(), cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        
        if (valid) {
            // 第二行：形状和颜色信息
            std::ostringstream shape_info;
            shape_info << "Shape: " << getShapeName(target.shape) 
                      << " | Color: " << (target.color == ColorMode::Red ? "RED" :
                                         target.color == ColorMode::Blue ? "BLUE" : "BOTH")
                      << " | Reg: " << std::fixed << std::setprecision(2) << target.regularity_score;
            cv::putText(frame, shape_info.str(), cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 2);
            
            // 第三行：方向和偏移
            std::ostringstream dir_info;
            dir_info << "fangxiang: " << getAngleDescription(deflection_rad) << " | Y: " << dy << "px";
            cv::putText(frame, dir_info.str(), cv::Point(10, 90), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,200,0), 2);
            
            // 底部：面积信息
            std::ostringstream area_info;
            area_info << "Area: " << std::fixed << std::setprecision(0) << target.area 
                     << " px (min: " << params_.min_area << ")";
            cv::putText(frame, area_info.str(), cv::Point(10, frame.rows - 50), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        } else {
            // 未检测到目标
            std::ostringstream no_target;
            no_target << "No valid target detected";
            cv::putText(frame, no_target.str(), cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
        }

        // 象限信息
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
        cv::putText(frame, quadrant_info.str(), cv::Point(10, 120), 
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
