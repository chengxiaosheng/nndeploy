# RTSP + YOLO Real-time Detection Demo

## 功能特性

- ✅ 支持RTSP实时视频流输入
- ✅ 支持YOLOv5/v6/v7/v8/v11模型
- ✅ 实时目标检测（人、车、动物等80类）
- ✅ 实时画框和标签显示
- ✅ FPS显示
- ✅ 支持GPU加速（CUDA/OpenVINO/TensorRT等）
- ✅ 支持Pipeline并行模式提升吞吐量
- ✅ 输出到文件或RTSP流

## 编译

```bash
cd /path/to/nndeploy/build
cmake ..
make nndeploy_demo_yolo_rtsp -j$(nproc)
```

## 使用方法

### 基本用法（RTSP输入 → 文件输出）

```bash
./nndeploy_demo_yolo_rtsp \
  --name nndeploy::detect::YoloGraph \
  --inference_type kInferenceTypeOnnxRuntime \
  --device_type kDeviceTypeCodeCuda:0 \
  --model_type kModelTypeOnnx \
  --is_path \
  --codec_flag kCodecFlagVideo \
  --parallel_type kParallelTypeSequential \
  --input_path "rtsp://192.168.1.100:554/stream" \
  --output_path "./output.mp4" \
  --model_value yolov8s.onnx
```

### GPU加速（推荐）

使用CUDA加速：
```bash
--device_type kDeviceTypeCodeCuda:0 \
--inference_type kInferenceTypeOnnxRuntime
```

使用TensorRT加速（最快）：
```bash
--device_type kDeviceTypeCodeCuda:0 \
--inference_type kInferenceTypeTensorRT
```

使用OpenVINO加速（Intel）：
```bash
--device_type kDeviceTypeCodeX86:0 \
--inference_type kInferenceTypeOpenVino
```

### Pipeline并行模式（提升吞吐量）

```bash
--parallel_type kParallelTypePipeline
```

这将让预处理、推理、后处理、编码同时工作在不同帧上，可提升40-60% FPS。

### RTSP流输出（推流）

如果需要将处理后的视频推送到RTSP服务器，可以使用第三方工具配合：

```bash
# 先输出到管道
./nndeploy_demo_yolo_rtsp ... --output_path - | \
  ffmpeg -i - -f rtsp rtsp://localhost:8554/output
```

## 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--name` | YOLO Graph名称 | `nndeploy::detect::YoloGraph` |
| `--inference_type` | 推理后端 | `kInferenceTypeOnnxRuntime`, `kInferenceTypeTensorRT` |
| `--device_type` | 设备类型 | `kDeviceTypeCodeCuda:0`, `kDeviceTypeCodeX86:0` |
| `--model_type` | 模型格式 | `kModelTypeOnnx` |
| `--is_path` | 模型路径标志 | （必须） |
| `--codec_flag` | 编解码器类型 | `kCodecFlagVideo` |
| `--parallel_type` | 并行模式 | `kParallelTypeSequential`, `kParallelTypePipeline` |
| `--input_path` | 输入RTSP URL | `rtsp://192.168.1.100:554/stream` |
| `--output_path` | 输出文件路径 | `./output.mp4` |
| `--model_value` | 模型文件路径 | `yolov8s.onnx` |

## YOLO模型准备

### 下载预训练模型

```bash
# YOLOv8s (推荐)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx

# YOLOv11s (最新)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.onnx

# YOLOv5s
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
```

### 自定义模型导出

```python
# YOLOv8/v11
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

## RTSP流测试

### 使用本地视频模拟RTSP流（用于测试）

```bash
# 安装rtsp-simple-server
wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.21.0/rtsp-simple-server_v0.21.0_linux_amd64.tar.gz
tar -xzf rtsp-simple-server_v0.21.0_linux_amd64.tar.gz
./rtsp-simple-server &

# 使用ffmpeg推送本地视频到RTSP服务器
ffmpeg -re -i input.mp4 -f rtsp rtsp://localhost:8554/stream

# 然后运行demo
./nndeploy_demo_yolo_rtsp --input_path "rtsp://localhost:8554/stream" ...
```

### 真实摄像头RTSP流

大多数IP摄像头支持RTSP，URL格式通常为：

```
rtsp://用户名:密码@IP地址:端口/path
```

示例：
- 海康威视：`rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101`
- 大华：`rtsp://admin:admin@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`

## 性能优化

### 1. 选择合适的模型

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| YOLOv8n | 最快 | 低 | 实时性要求高 |
| YOLOv8s | 快 | 中 | **推荐** |
| YOLOv8m | 中 | 高 | 精度要求高 |
| YOLOv11s | 快 | 中高 | 最新版本 |

### 2. 使用TensorRT加速

```bash
# 先转换为TensorRT引擎
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.trt --fp16

# 使用TensorRT推理
--inference_type kInferenceTypeTensorRT \
--model_value yolov8s.trt
```

### 3. 启用Pipeline并行

```bash
--parallel_type kParallelTypePipeline
```

### 4. 调整图像分辨率

如果RTSP流分辨率很高，可以通过修改YoloPostParam来调整：
- 默认：640x640
- 建议：根据实际场景选择320/416/640/1280

## 故障排除

### RTSP连接失败

```bash
# 测试RTSP流是否可用
ffplay rtsp://192.168.1.100:554/stream
```

### GPU内存不足

```bash
# 使用更小的模型
--model_value yolov8n.onnx

# 或减小输入分辨率
```

### 帧率低

1. 检查GPU是否正常工作（nvidia-smi）
2. 使用Pipeline模式（--parallel_type kParallelTypePipeline）
3. 使用TensorRT加速
4. 使用更小的模型

## 示例输出

```
===== RTSP + YOLO Real-time Detection =====
Input (RTSP): rtsp://192.168.1.100:554/stream
Output: ./output.mp4
Model: yolov8s.onnx
Device: CUDA:0
Inference: OnnxRuntime

Starting inference loop...
Press Ctrl+C to stop
Frame 30: Detected 5 objects
Frame 60: Detected 3 objects
Frame 90: Detected 7 objects
...

TimeProfiler: rtsp_yolo_demo
-------------------------------------------------------------------------------------------
name                         call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-------------------------------------------------------------------------------------------
graph->init()                1           450.123            450.123            0.000
graph->run                   1           15234.567          15234.567          0.000
decode_node run()            500         1234.567           2.469              0.000
preprocess run()             500         456.789            0.914              0.000
infer run()                  500         8901.234           17.802             0.000
postprocess run()            500         1234.567           2.469              0.000
vis_node run()               500         890.123            1.780              0.000
encode_node run()            500         2345.678           4.691              0.000
-------------------------------------------------------------------------------------------

Processed 500 frames
Demo finished successfully!
```

## 扩展开发

### 自定义检测类别过滤

修改`VisDetection::run()`中的绘制逻辑：

```cpp
// 只检测人（class_id == 0）
if (class_id == 0) {
  // 绘制框...
}
```

### 添加区域入侵检测

```cpp
// 定义感兴趣区域
cv::Rect roi(100, 100, 400, 300);

// 检查检测框是否在ROI内
for (const auto &bbox : detect_result->bbox_) {
  cv::Rect det_rect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]);
  if ((roi & det_rect).area() > 0) {
    NNDEPLOY_LOGI("Object detected in ROI!");
  }
}
```

### 添加计数功能

```cpp
static std::map<int, int> class_count;
for (int class_id : detect_result->label_id_) {
  class_count[class_id]++;
}
```

## 参考资料

- nndeploy文档：https://nndeploy-zh.readthedocs.io/
- YOLOv8：https://github.com/ultralytics/ultralytics
- RTSP协议：https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol
