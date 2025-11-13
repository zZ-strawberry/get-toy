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

// 计算轮廓规则性得分（越高越规则）
struct ShapeMetrics {
    double rectangularity;  // 矩形度：轮廓面积 / 外接矩形面积
    double squareness;      // 正方形度：min(w,h) / max(w,h)
    double convexity;       // 凸性：轮廓面积 / 凸包面积
    double compactness;     // 紧凑度：4π·面积 / 周长²
    double score;           // 综合得分
    bool isSquare;          // 是否为正方形
    bool isRectangle;       // 是否为矩形
};

ShapeMetrics evaluateShape(const vector<Point>& contour) {
    ShapeMetrics metrics;
    
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    
    // 多边形逼近
    vector<Point> approx;
    approxPolyDP(contour, approx, 0.04 * perimeter, true);
    int vertices = approx.size();
    
    // 计算外接矩形
    Rect boundingBox = boundingRect(contour);
    double boxArea = boundingBox.width * boundingBox.height;
    double aspectRatio = std::min(boundingBox.width, boundingBox.height) / 
                        (double)std::max(boundingBox.width, boundingBox.height);
    
    // 矩形度
    metrics.rectangularity = boxArea > 0 ? area / boxArea : 0;
    
    // 正方形度
    metrics.squareness = aspectRatio;
    
    // 凸性
    vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = contourArea(hull);
    metrics.convexity = hullArea > 0 ? area / hullArea : 0;
    
    // 紧凑度
    metrics.compactness = perimeter > 0 ? (4 * M_PI * area) / (perimeter * perimeter) : 0;
    
    // 判断是否为矩形（4个顶点）
    metrics.isRectangle = (vertices == 4);
    
    // 判断是否为正方形（4个顶点 + 宽高比接近1）
    metrics.isSquare = metrics.isRectangle && (aspectRatio > 0.85);  // 容差15%
    
    // 综合得分（正方形权重最高）
    if (metrics.isSquare) {
        metrics.score = 100.0 + metrics.squareness * 50;  // 正方形基础分100
    } else if (metrics.isRectangle) {
        metrics.score = 80.0 + metrics.rectangularity * 20;  // 矩形基础分80
    } else {
        // 其他形状按规则性评分
        metrics.score = metrics.rectangularity * 25 + 
                       metrics.convexity * 25 + 
                       metrics.compactness * 25 +
                       metrics.squareness * 25;
    }
    
    return metrics;
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

        // 查找轮廓
        vector<vector<Point>> contours;
        findContours(processed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // 多级筛选：正方形 > 矩形 > 最规则图形
        bool found = false;
        float angle = 0.0f;
        float y_offset_px = 0.0f; 
        Point2f center(0, 0);
        vector<Point> bestContour;
        ShapeMetrics bestMetrics;
        bestMetrics.score = -1;
        bool hasSquare = false;
        bool hasRectangle = false;

        for (const auto& contour : contours) {
            double area = contourArea(contour);
            
            // 最小面积过滤
            if (area < absolute_min_area_) continue;
            
            // 评估形状
            ShapeMetrics metrics = evaluateShape(contour);
            
            // 第一优先级：找到正方形
            if (metrics.isSquare) {
                if (!hasSquare || area > contourArea(bestContour)) {
                    bestContour = contour;
                    bestMetrics = metrics;
                    found = true;
                    hasSquare = true;
                }
            }
            // 第二优先级：没有正方形时找矩形
            else if (!hasSquare && metrics.isRectangle) {
                if (!hasRectangle || area > contourArea(bestContour)) {
                    bestContour = contour;
                    bestMetrics = metrics;
                    found = true;
                    hasRectangle = true;
                }
            }
            // 第三优先级：没有矩形时找最规则的图形
            else if (!hasSquare && !hasRectangle) {
                if (metrics.score > bestMetrics.score) {
                    bestContour = contour;
                    bestMetrics = metrics;
                    found = true;
                }
            }
        }

        // 画面中心
        Point2f imageCenter(frame.cols / 2.0f, frame.rows / 2.0f);

        if (found) {
            // 计算中心点与角度
            center = getContourCenter(bestContour);
            float dx = center.x - imageCenter.x;
            float dy = imageCenter.y - center.y;
            angle = std::atan2(dx, dy);
            y_offset_px = static_cast<float>(std::lround(center.y - imageCenter.y));

            // 可视化 - 根据形状类型使用不同颜色
            Scalar color = bestMetrics.isSquare ? Scalar(0, 255, 0) :      // 绿色：正方形
                          bestMetrics.isRectangle ? Scalar(255, 255, 0) :  // 青色：矩形
                          Scalar(0, 165, 255);                             // 橙色：其他
            
            drawContours(frame, vector<vector<Point>>{bestContour}, -1, color, 2);
            Rect boundingBox = boundingRect(bestContour);
            rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);
            circle(frame, center, 8, Scalar(0, 0, 255), -1);
            line(frame, imageCenter, center, Scalar(255, 255, 0), 2);

            // 显示信息
            char text[200];
            const char* shapeType = bestMetrics.isSquare ? "Square" : 
                                   bestMetrics.isRectangle ? "Rectangle" : "Shape";
            sprintf(text, "%s | Angle: %.2f rad | Score: %.1f", shapeType, angle, bestMetrics.score);
            putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            
            sprintf(text, "Rect:%.2f Sqr:%.2f Conv:%.2f Comp:%.2f", 
                    bestMetrics.rectangularity, bestMetrics.squareness, 
                    bestMetrics.convexity, bestMetrics.compactness);
            putText(frame, text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }

        // 发布数据
        publishData(found, angle, y_offset_px);

        // 十字线
        circle(frame, imageCenter, 5, Scalar(255, 0, 0), -1);
        line(frame, Point(imageCenter.x, 0), Point(imageCenter.x, frame.rows), Scalar(255, 0, 0), 1);
        line(frame, Point(0, imageCenter.y), Point(frame.cols, imageCenter.y), Scalar(255, 0, 0), 1);

        imshow("Black Object Detection", frame);
        waitKey(1);
    }
    
    void publishData(bool hasTarget, float angle, float yOffsetPx) {
        // 状态: 浮点数 0.00 或 2.00
        auto status_msg = std_msgs::msg::Float32();
        float status_val = hasTarget ? 2.0f : 0.0f;
        status_msg.data = std::round(status_val * 100.0f) / 100.0f;
        status_pub_->publish(status_msg);

        // 角度: 浮点数（弧度），保留两位小数
        auto angle_msg = std_msgs::msg::Float32();
        angle_msg.data = std::round(angle * 100.0f) / 100.0f;
        angle_pub_->publish(angle_msg);

        // 偏移: 浮点数（像素），保留两位小数
        auto offset_msg = std_msgs::msg::Float32();
        float offset_val = static_cast<float>(yOffsetPx);
        offset_msg.data = std::round(offset_val * 100.0f) / 100.0f;
        offset_pub_->publish(offset_msg);
    }
    
    VideoCapture cap_;
    
    int v_max_ = 58;     // V通道上限（暗）
    int s_max_ = 255;    // S通道上限
    int blur_size_ = 2;  // 实际大小 = blur_size_ * 2 + 1
    int min_area_ = 10;  // 实际面积 = min_area_ * 100
    int morph_size_ = 3; // 形态学核大小
    
    // 新增：绝对最小面积阈值
    double absolute_min_area_ = 5000.0;  // 可调整，过滤噪点
    
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
