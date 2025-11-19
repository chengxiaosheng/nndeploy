#!/bin/bash

# RTSP + YOLO Real-time Detection Quick Start Script

set -e

echo "================================================"
echo "  RTSP + YOLO Real-time Detection Quick Start"
echo "================================================"
echo ""

# 检查是否在build目录
if [ ! -f "CMakeCache.txt" ]; then
    echo "Error: Please run this script from the build directory"
    echo "Usage: cd /path/to/nndeploy/build && bash ../demo/yolo_rtsp/quickstart.sh"
    exit 1
fi

# 编译demo
echo "[1/4] Compiling demo..."
make nndeploy_demo_yolo_rtsp -j$(nproc)
echo "✓ Compilation successful!"
echo ""

# 检查YOLO模型
MODEL_DIR="$(pwd)/models"
YOLO_MODEL="$MODEL_DIR/yolov8s.onnx"

if [ ! -f "$YOLO_MODEL" ]; then
    echo "[2/4] Downloading YOLOv8s model..."
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx || \
    {
        echo "Failed to download model. Please download manually:"
        echo "wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx -O $YOLO_MODEL"
        exit 1
    }
    cd -
    echo "✓ Model downloaded!"
else
    echo "[2/4] YOLOv8s model found: $YOLO_MODEL"
fi
echo ""

# 准备测试视频或RTSP流
echo "[3/4] Preparing input source..."
read -p "Enter RTSP URL or video file path (default: use sample video): " INPUT_PATH

if [ -z "$INPUT_PATH" ]; then
    # 下载示例视频
    SAMPLE_VIDEO="$MODEL_DIR/sample_video.mp4"
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        echo "Downloading sample video..."
        wget -q --show-progress https://github.com/nndeploy/nndeploy/releases/download/v0.1.0/sample_video.mp4 -O "$SAMPLE_VIDEO" || \
        {
            echo "Failed to download sample video. Please provide your own video or RTSP URL."
            exit 1
        }
    fi
    INPUT_PATH="$SAMPLE_VIDEO"
    echo "Using sample video: $INPUT_PATH"
else
    echo "Using input: $INPUT_PATH"
fi
echo ""

# 设置输出路径
OUTPUT_PATH="$(pwd)/output_yolo_rtsp.mp4"

# 询问设备类型
echo "[4/4] Select device type:"
echo "  1) CUDA (GPU) - Recommended"
echo "  2) CPU"
read -p "Enter choice [1-2] (default: 1): " DEVICE_CHOICE

case "$DEVICE_CHOICE" in
    2)
        DEVICE_TYPE="kDeviceTypeCodeX86:0"
        INFERENCE_TYPE="kInferenceTypeOnnxRuntime"
        echo "Selected: CPU with ONNXRuntime"
        ;;
    *)
        DEVICE_TYPE="kDeviceTypeCodeCuda:0"
        INFERENCE_TYPE="kInferenceTypeOnnxRuntime"
        echo "Selected: CUDA GPU with ONNXRuntime"
        ;;
esac
echo ""

# 运行demo
echo "================================================"
echo "  Starting YOLO RTSP Demo..."
echo "================================================"
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Model:  $YOLO_MODEL"
echo "Device: $DEVICE_TYPE"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

./nndeploy_demo_yolo_rtsp \
    --name nndeploy::detect::YoloGraph \
    --inference_type "$INFERENCE_TYPE" \
    --device_type "$DEVICE_TYPE" \
    --model_type kModelTypeOnnx \
    --is_path \
    --codec_flag kCodecFlagVideo \
    --parallel_type kParallelTypeSequential \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_value "$YOLO_MODEL"

echo ""
echo "================================================"
echo "  Demo completed successfully!"
echo "================================================"
echo "Output saved to: $OUTPUT_PATH"
echo ""
echo "To view the output:"
echo "  ffplay $OUTPUT_PATH"
echo "  # or"
echo "  vlc $OUTPUT_PATH"
echo ""
