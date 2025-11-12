# 形状识别追踪系统说明

## 功能概述

本系统可以识别和追踪特定颜色（红色或蓝色）的特定形状（三角形、矩形、圆形）。

## 工作逻辑

### 1. 颜色过滤（必须满足）
- **红色**：HSV范围 (0-10, 120-255, 70-255) 和 (170-180, 120-255, 70-255)
- **蓝色**：HSV范围 (100-140, 120-255, 60-255)
- **重要**：如果画面中没有红色或蓝色，则不会追踪任何目标

### 2. 形状识别（优先级排序）
系统按以下优先级选择目标：

#### 优先级 1：匹配指定形状
如果检测到以下任一形状，将优先追踪：
- **三角形**：轮廓近似后有3个顶点
- **矩形**：轮廓近似后有4个顶点，且矩形度 ≥ 0.75
- **圆形**：圆形度 ≥ 0.7，且顶点数 > 6

#### 优先级 2：最规整的轮廓
如果没有匹配的特定形状，系统会选择**最规整的轮廓**（规整度分数最高）。

规整度评分基于：
- **圆形度**：4π × 面积 / 周长²（完美圆形为1）
- **凸性**：轮廓面积 / 凸包面积（完美凸形为1）
- 综合分数 = 圆形度 × 0.6 + 凸性 × 0.4

### 3. 其他过滤条件
- 面积必须在 `min_area` 和 `max_area` 之间
- 颜色必须是红色或蓝色（非红非蓝的物体被忽略）

## 配置参数

### 形状识别开关
```yaml
enable_shape_detection: true    # 启用/禁用形状识别
track_triangle: true            # 是否追踪三角形
track_rectangle: true           # 是否追踪矩形
track_circle: true              # 是否追踪圆形
```

### 识别阈值
```yaml
min_regularity: 0.5             # 最小规整度阈值 (0-1)
circularity_threshold: 0.7      # 圆形判定阈值 (0-1)
rectangularity_threshold: 0.75  # 矩形判定阈值 (0-1)
```

### 基本检测参数
```yaml
min_area: 2000                  # 最小面积（像素）
max_area: 50000                 # 最大面积（像素）
color_mode: "both"              # red/blue/both
```

## 界面显示

运行时界面会显示：
- **Shape**：识别到的形状类型（Triangle/Rectangle/Circle/Unknown）
- **Color**：检测到的颜色（RED/BLUE/BOTH）
- **Reg**：规整度分数（0.00-1.00）
- **Area**：轮廓面积（像素）
- 轮廓颜色：红色目标用红色边框，蓝色目标用蓝色边框

## 使用场景

### 场景 1：只追踪特定形状
```yaml
enable_shape_detection: true
track_triangle: true
track_rectangle: false
track_circle: false
```
结果：只追踪三角形，忽略其他形状

### 场景 2：追踪所有形状或最规整轮廓
```yaml
enable_shape_detection: true
track_triangle: true
track_rectangle: true
track_circle: true
```
结果：优先追踪三角形/矩形/圆形，如果都没有则追踪最规整的轮廓

### 场景 3：只追踪最规整的轮廓
```yaml
enable_shape_detection: true
track_triangle: false
track_rectangle: false
track_circle: false
```
结果：追踪规整度最高的轮廓（无论什么形状）

### 场景 4：禁用形状识别（向后兼容）
```yaml
enable_shape_detection: false
```
结果：追踪面积最大的轮廓（旧版行为）

## 调试技巧

1. **观察规整度分数**：如果目标形状不规则，规整度会很低（< 0.5）
2. **调整阈值**：
   - 增大 `circularity_threshold` 使圆形判定更严格
   - 增大 `rectangularity_threshold` 使矩形判定更严格
   - 降低 `min_regularity` 允许追踪不太规整的轮廓
3. **颜色调试**：使用 `publish_mask: true` 查看颜色掩码

## 注意事项

1. **颜色是强制要求**：无红色或蓝色 = 不追踪
2. **光照影响**：HSV颜色范围在不同光照下可能需要调整
3. **性能**：形状识别会略微增加计算量，但在现代处理器上影响很小
