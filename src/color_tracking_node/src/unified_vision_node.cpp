#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace cv;

// 视觉模式
enum class VisionMode {
    DISABLED = 0,         // 停止/禁用
    COLOR_TRACKING = 1,   // 彩色跟踪
    BASKET_DETECTION = 2  // 黑色篮筐检测
};

// ===== 彩色跟踪相关 =====
static void makeMaskHSV(const Mat& hsv, Mat& maskOut) {
    Mat maskRed, maskBlue, lower, upper;
    
    // 红色
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), lower);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), upper);
    bitwise_or(lower, upper, maskRed);
    
    // 蓝色（排除黑色）
    Mat blue_mask_primary, black_mask;
    inRange(hsv, Scalar(100, 150, 80), Scalar(130, 255, 255), blue_mask_primary);
    inRange(hsv, Scalar(0, 0, 0), Scalar(180, 50, 50), black_mask);
    bitwise_and(blue_mask_primary, Scalar(255) - black_mask, maskBlue);
    
    bitwise_or(maskRed, maskBlue, maskOut);
    
    // 形态学处理
    Mat kernel_open = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    morphologyEx(maskOut, maskOut, MORPH_OPEN, kernel_open, Point(-1,-1), 2);
    morphologyEx(maskOut, maskOut, MORPH_CLOSE, kernel_close, Point(-1,-1), 2);
    threshold(maskOut, maskOut, 127, 255, THRESH_BINARY);
}

static double calculateRegularityScore(const std::vector<Point>& contour) {
    if (contour.size() < 3) return 0.0;
    
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    if (perimeter < 1e-5) return 0.0;
    
    double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    
    std::vector<Point> hull;
    convexHull(contour, hull);
    double hull_area = contourArea(hull);
    double convexity = (hull_area > 0) ? (area / hull_area) : 0.0;
    
    return std::min(1.0, (circularity * 0.6 + convexity * 0.4));
}

// ===== 黑色篮筐检测相关 =====
struct ShapeMetrics {
    double rectangularity;
    double squareness;
    double convexity;
    double compactness;
    double score;
    bool isSquare;
    bool isRectangle;
};

static ShapeMetrics evaluateShape(const std::vector<Point>& contour) {
    ShapeMetrics metrics;
    
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    
    std::vector<Point> approx;
    approxPolyDP(contour, approx, 0.04 * perimeter, true);
    int vertices = approx.size();
    
    Rect boundingBox = boundingRect(contour);
    double boxArea = boundingBox.width * boundingBox.height;
    double aspectRatio = std::min(boundingBox.width, boundingBox.height) / 
                        (double)std::max(boundingBox.width, boundingBox.height);
    
    metrics.rectangularity = boxArea > 0 ? area / boxArea : 0;
    metrics.squareness = aspectRatio;
    
    std::vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = contourArea(hull);
    metrics.convexity = hullArea > 0 ? area / hullArea : 0;
    metrics.compactness = perimeter > 0 ? (4 * M_PI * area) / (perimeter * perimeter) : 0;
    
    metrics.isRectangle = (vertices == 4);
    metrics.isSquare = metrics.isRectangle && (aspectRatio > 0.85);
    
    if (metrics.isSquare) {
        metrics.score = 100.0 + metrics.squareness * 50;
    } else if (metrics.isRectangle) {
        metrics.score = 80.0 + metrics.rectangularity * 20;
    } else {
        metrics.score = metrics.rectangularity * 25 + metrics.convexity * 25 + 
                       metrics.compactness * 25 + metrics.squareness * 25;
    }
    
    return metrics;
}

// ===== 统一视觉节点 =====
class UnifiedVisionNode : public rclcpp::Node {
public:
    UnifiedVisionNode() : Node("unified_vision_node"), 
                         current_mode_(VisionMode::COLOR_TRACKING),
                         last_mode_(VisionMode::COLOR_TRACKING) {
        // 订阅模式切换命令
        mode_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/task_command", 10,
            std::bind(&UnifiedVisionNode::modeCallback, this, std::placeholders::_1));
        
        // 统一发布器 - 所有模式使用相同话题和格式
        color_result_pub_ = this->create_publisher<std_msgs::msg::String>("color_tracking/result", 10);
        
        // 打开摄像头
        cap_.open(0);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开摄像头");
            rclcpp::shutdown();
            return;
        }
        
        cap_.set(CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap_.set(CAP_PROP_FPS, 30);
        
        // 处理定时器 30Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&UnifiedVisionNode::processFrame, this));
        
        RCLCPP_INFO(this->get_logger(), "统一视觉节点已启动 - 默认模式: 彩色跟踪");
    }
    
    ~UnifiedVisionNode() {
        if (cap_.isOpened()) cap_.release();
        destroyAllWindows();
    }

private:
    void modeCallback(const std_msgs::msg::Int32::SharedPtr msg) {
        int mode_cmd = msg->data;
        VisionMode old_mode = current_mode_;
        
        if (mode_cmd == 0) {
            current_mode_ = VisionMode::DISABLED;
            RCLCPP_INFO(this->get_logger(), "🛑 视觉处理已停止");
        } else if (mode_cmd == 1) {
            current_mode_ = VisionMode::COLOR_TRACKING;
            RCLCPP_INFO(this->get_logger(), "切换到: 彩色跟踪模式");
        } else if (mode_cmd == 2) {
            current_mode_ = VisionMode::BASKET_DETECTION;
            RCLCPP_INFO(this->get_logger(), "切换到: 黑色篮筐检测模式");
        }
        
        // 切换模式时销毁所有旧窗口，避免窗口残留
        if (old_mode != current_mode_) {
            destroyAllWindows();
            last_mode_ = old_mode;  // 记录上次模式
            RCLCPP_DEBUG(this->get_logger(), "已清理旧窗口");
        }
    }
    
    void processFrame() {
        Mat frame;
        cap_ >> frame;
        if (frame.empty()) return;
        
        // 只在模式切换时清理窗口，避免每帧都销毁不存在的窗口
        if (current_mode_ != last_mode_) {
            destroyAllWindows();
            last_mode_ = current_mode_;
        }
        
        if (current_mode_ == VisionMode::DISABLED) {
            // 停止模式：只显示画面，不做任何处理
            putText(frame, "Vision DISABLED", Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            imshow("Unified Vision - Disabled", frame);
        } else if (current_mode_ == VisionMode::COLOR_TRACKING) {
            processColorTracking(frame);
        } else if (current_mode_ == VisionMode::BASKET_DETECTION) {
            processBasketDetection(frame);
        }
        
        waitKey(1);
    }
    
    // 彩色跟踪处理
    void processColorTracking(Mat& frame) {
        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        makeMaskHSV(hsv, mask);
        
        std::vector<std::vector<Point>> contours;
        Mat mask_copy = mask.clone();
        findContours(mask_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        bool found = false;
        Point2f target_center(0, 0);
        double best_score = -1.0;
        std::vector<Point> best_contour;
        
        for (const auto& c : contours) {
            double area = contourArea(c);
            if (area < 2000 || area > 50000) continue;
            
            double score = calculateRegularityScore(c);
            if (score > best_score) {
                best_score = score;
                best_contour = c;
                
                Moments m = moments(c);
                if (std::abs(m.m00) > 1e-5) {
                    target_center = Point2f(m.m10 / m.m00, m.m01 / m.m00);
                    found = true;
                }
            }
        }
        
        Point2f image_center(frame.cols / 2.0f, frame.rows / 2.0f);
        double deflection_rad = 0.0;
        int dy = 0;
        
        if (found) {
            float dx = target_center.x - image_center.x;
            float dy_f = image_center.y - target_center.y;
            deflection_rad = std::atan2(dx, dy_f);
            dy = static_cast<int>(std::round(target_center.y - image_center.y));
            
            // 可视化
            drawContours(frame, std::vector<std::vector<Point>>{best_contour}, -1, Scalar(0, 255, 0), 2);
            circle(frame, target_center, 6, Scalar(0, 0, 255), -1);
            arrowedLine(frame, image_center, target_center, Scalar(255, 255, 0), 2);
        }
        
        // 发布结果
        auto msg = std_msgs::msg::String();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << (found ? 1.00f : 0.00f) << "," << deflection_rad << "," << dy;
        msg.data = oss.str();
        color_result_pub_->publish(msg);
        
        // 显示信息
        char text[200];
        sprintf(text, "COLOR_TRACKING | valid=%d | rad=%.2f | dy=%d", found, deflection_rad, dy);
        putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        // 绘制中心十字
        line(frame, Point(image_center.x, 0), Point(image_center.x, frame.rows), Scalar(255, 0, 0), 1);
        line(frame, Point(0, image_center.y), Point(frame.cols, image_center.y), Scalar(255, 0, 0), 1);
        
        imshow("Unified Vision - Color Tracking", frame);
    }
    
    // 黑色篮筐检测处理
    void processBasketDetection(Mat& frame) {
        Mat hsv, binary, processed;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        
        int blur_size = 5;
        GaussianBlur(hsv, hsv, Size(blur_size, blur_size), 0);
        
        // HSV黑色检测
        Scalar lower_black(0, 0, 0);
        Scalar upper_black(180, 255, 58);
        inRange(hsv, lower_black, upper_black, binary);
        
        // 形态学处理
        Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
        morphologyEx(binary, processed, MORPH_CLOSE, kernel);
        morphologyEx(processed, processed, MORPH_OPEN, kernel);
        
        std::vector<std::vector<Point>> contours;
        findContours(processed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        bool found = false;
        float angle = 0.0f;
        float y_offset_px = 0.0f;
        Point2f center(0, 0);
        std::vector<Point> bestContour;
        ShapeMetrics bestMetrics;
        bestMetrics.score = -1;
        bool hasSquare = false;
        bool hasRectangle = false;
        
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 5000.0) continue;
            
            ShapeMetrics metrics = evaluateShape(contour);
            
            // 只接受正方形或矩形
            if (!metrics.isRectangle) continue;
            
            if (metrics.isSquare) {
                if (!hasSquare || area > contourArea(bestContour)) {
                    bestContour = contour;
                    bestMetrics = metrics;
                    found = true;
                    hasSquare = true;
                }
            } else {
                // 是矩形但不是正方形
                if (!hasSquare && (!hasRectangle || area > contourArea(bestContour))) {
                    bestContour = contour;
                    bestMetrics = metrics;
                    found = true;
                    hasRectangle = true;
                }
            }
        }
        
        Point2f imageCenter(frame.cols / 2.0f, frame.rows / 2.0f);
        
        if (found) {
            Moments m = moments(bestContour);
            if (m.m00 != 0) {
                center = Point2f(m.m10 / m.m00, m.m01 / m.m00);
            }
            
            float dx = center.x - imageCenter.x;
            float dy = imageCenter.y - center.y;
            angle = std::atan2(dx, dy);
            y_offset_px = static_cast<float>(std::lround(center.y - imageCenter.y));
            
            // 可视化
            Scalar color = bestMetrics.isSquare ? Scalar(0, 255, 0) :
                          bestMetrics.isRectangle ? Scalar(255, 255, 0) : Scalar(0, 165, 255);
            drawContours(frame, std::vector<std::vector<Point>>{bestContour}, -1, color, 2);
            Rect boundingBox = boundingRect(bestContour);
            rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);
            circle(frame, center, 8, Scalar(0, 0, 255), -1);
            line(frame, imageCenter, center, Scalar(255, 255, 0), 2);
            
            char text[200];
            const char* shapeType = bestMetrics.isSquare ? "Square" : 
                                   bestMetrics.isRectangle ? "Rectangle" : "Shape";
            sprintf(text, "%s | Angle: %.2f rad | Score: %.1f", shapeType, angle, bestMetrics.score);
            putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        
        // 统一发布格式：与彩色跟踪相同 "status,angle,offset"
        // status: 2.00(检测到) / 0.00(未检测到)
        auto msg = std_msgs::msg::String();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        float status_value = found ? 2.00f : 0.00f;
        oss << status_value << "," << angle << "," << static_cast<int>(y_offset_px);
        msg.data = oss.str();
        color_result_pub_->publish(msg);
        
        // 显示模式信息
        char mode_text[100];
        sprintf(mode_text, "BASKET_DETECTION | found=%d | status=%.2f", found, status_value);
        putText(frame, mode_text, Point(10, frame.rows - 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
        
        // 十字线
        circle(frame, imageCenter, 5, Scalar(255, 0, 0), -1);
        line(frame, Point(imageCenter.x, 0), Point(imageCenter.x, frame.rows), Scalar(255, 0, 0), 1);
        line(frame, Point(0, imageCenter.y), Point(frame.cols, imageCenter.y), Scalar(255, 0, 0), 1);
        
        imshow("Unified Vision - Basket Detection", frame);
    }
    
    VideoCapture cap_;
    VisionMode current_mode_;
    VisionMode last_mode_;  // 跟踪上一次的模式，用于检测模式切换
    
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mode_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr color_result_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UnifiedVisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
