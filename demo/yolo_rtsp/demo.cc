/**
 * nndeploy RTSP + YOLO Real-time Detection Demo
 * Features:
 * - RTSP stream input with FFmpeg GPU hardware decode
 * - Real-time YOLO object detection
 * - Bounding box visualization
 * - Real-time display with cv::imshow
 * - Optional file output
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/detect/yolo/yolo.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

// FFmpeg headers for direct hardware decoding
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace nndeploy;

// 全局标志：是否启用实时显示
static bool g_enable_display = true;

/**
 * @brief RTSP解码节点 - 使用FFmpeg GPU硬件解码
 */
class RtspVideoDecode : public codec::Decode {
 public:
  RtspVideoDecode(const std::string &name, dag::Edge *output, bool use_gpu = true)
      : codec::Decode(name, {}, {output}, base::kCodecFlagVideo), use_gpu_(use_gpu) {
    key_ = "nndeploy::demo::RtspVideoDecode";
    desc_ = "RTSP Video Decode with FFmpeg GPU hardware decode";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }

  virtual ~RtspVideoDecode() {
    deinit();
  }

  virtual base::Status init() override {
    return base::kStatusCodeOk;
  }

  virtual base::Status deinit() override {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
      sws_ctx_ = nullptr;
    }
    if (sw_frame_) {
      av_frame_free(&sw_frame_);
    }
    if (frame_) {
      av_frame_free(&frame_);
    }
    if (packet_) {
      av_packet_free(&packet_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
      avformat_close_input(&fmt_ctx_);
    }
    if (hw_device_ctx_) {
      av_buffer_unref(&hw_device_ctx_);
    }

    // 兼容旧的OpenCV方式
    if (cap_ != nullptr && cap_->isOpened()) {
      cap_->release();
      delete cap_;
      cap_ = nullptr;
    }

    return base::kStatusCodeOk;
  }

  virtual base::Status setPath(const std::string &path) override {
    path_ = path;
    bool is_rtsp = (path_.find("rtsp://") == 0);

    if (!use_gpu_ || !is_rtsp) {
      // 回退到OpenCV CPU解码（用于本地文件或禁用GPU时）
      NNDEPLOY_LOGI("Using OpenCV CPU decode for: %s", path_.c_str());
      return setupOpenCVDecode(path_);
    }

    // ========== FFmpeg GPU硬件解码流程 ==========
    NNDEPLOY_LOGI("Initializing FFmpeg GPU hardware decode for RTSP...");

    // 1. 打开输入流
    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);  // 强制TCP
    av_dict_set(&opts, "stimeout", "5000000", 0);     // 5秒超时

    int ret = avformat_open_input(&fmt_ctx_, path_.c_str(), nullptr, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(ret, errbuf, sizeof(errbuf));
      NNDEPLOY_LOGE("Cannot open input '%s': %s", path_.c_str(), errbuf);
      return base::kStatusCodeErrorInvalidParam;
    }

    // 2. 获取流信息
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
      NNDEPLOY_LOGE("Cannot find stream information");
      return base::kStatusCodeErrorInvalidParam;
    }

    // 3. 查找视频流
    video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
      NNDEPLOY_LOGE("Cannot find video stream");
      return base::kStatusCodeErrorInvalidParam;
    }

    AVStream *video_stream = fmt_ctx_->streams[video_stream_idx_];

    // 4. 查找硬件解码器（h264_cuvid）
    const AVCodec *decoder = avcodec_find_decoder_by_name("h264_cuvid");
    if (!decoder) {
      NNDEPLOY_LOGW("h264_cuvid decoder not found, falling back to CPU");
      use_gpu_ = false;
      avformat_close_input(&fmt_ctx_);
      return setupOpenCVDecode(path_);
    }

    codec_ctx_ = avcodec_alloc_context3(decoder);
    if (!codec_ctx_) {
      NNDEPLOY_LOGE("Failed to allocate codec context");
      return base::kStatusCodeErrorOutOfMemory;
    }

    // 5. 设置解码器参数
    if (avcodec_parameters_to_context(codec_ctx_, video_stream->codecpar) < 0) {
      NNDEPLOY_LOGE("Failed to copy codec parameters");
      return base::kStatusCodeErrorInvalidParam;
    }

    // 6. 创建CUDA硬件设备上下文
    ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, "0", nullptr, 0);
    if (ret < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(ret, errbuf, sizeof(errbuf));
      NNDEPLOY_LOGW("Failed to create CUDA device: %s, falling back to CPU", errbuf);
      use_gpu_ = false;
      avcodec_free_context(&codec_ctx_);
      avformat_close_input(&fmt_ctx_);
      return setupOpenCVDecode(path_);
    }

    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

    // 7. 打开解码器
    if (avcodec_open2(codec_ctx_, decoder, nullptr) < 0) {
      NNDEPLOY_LOGE("Cannot open video decoder");
      return base::kStatusCodeErrorInvalidParam;
    }

    // 8. 分配帧缓冲
    frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    if (!frame_ || !sw_frame_ || !packet_) {
      NNDEPLOY_LOGE("Cannot allocate frames/packet");
      return base::kStatusCodeErrorOutOfMemory;
    }

    // 9. 获取视频信息
    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;
    fps_ = av_q2d(video_stream->avg_frame_rate);
    if (fps_ <= 0) fps_ = 25.0;

    NNDEPLOY_LOGI("✅ FFmpeg GPU decode initialized successfully!");
    NNDEPLOY_LOGI("   Resolution: %dx%d, FPS: %.2f, Decoder: %s",
                  width_, height_, fps_, decoder->name);

    index_ = 0;
    return base::kStatusCodeOk;
  }

  // 辅助函数：使用OpenCV进行CPU解码（回退方案）
  base::Status setupOpenCVDecode(const std::string& path) {
    if (cap_ != nullptr) {
      cap_->release();
      delete cap_;
    }

    cap_ = new cv::VideoCapture();
    cap_->set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000);
    cap_->set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000);

    bool is_rtsp = (path.find("rtsp://") == 0);
    if (is_rtsp) {
      std::string rtsp_url = path + "?tcp";
      if (!cap_->open(rtsp_url, cv::CAP_FFMPEG) && !cap_->open(path, cv::CAP_FFMPEG)) {
        NNDEPLOY_LOGE("Failed to open RTSP stream: %s", path.c_str());
        delete cap_;
        cap_ = nullptr;
        return base::kStatusCodeErrorInvalidParam;
      }
    } else {
      if (!cap_->open(path)) {
        NNDEPLOY_LOGE("Failed to open video file: %s", path.c_str());
        delete cap_;
        cap_ = nullptr;
        return base::kStatusCodeErrorInvalidParam;
      }
    }

    if (!cap_->isOpened()) {
      NNDEPLOY_LOGE("VideoCapture not opened");
      delete cap_;
      cap_ = nullptr;
      return base::kStatusCodeErrorInvalidParam;
    }

    width_ = (int)cap_->get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_->get(cv::CAP_PROP_FRAME_HEIGHT);
    fps_ = cap_->get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 25.0;

    NNDEPLOY_LOGI("OpenCV CPU decode ready: %dx%d @ %.2f FPS", width_, height_, fps_);
    index_ = 0;
    return base::kStatusCodeOk;
  }

  virtual base::Status run() override {
    if (use_gpu_ && fmt_ctx_) {
      // FFmpeg GPU硬件解码路径
      return runFFmpegDecode();
    } else if (cap_ && cap_->isOpened()) {
      // OpenCV CPU解码路径
      return runOpenCVDecode();
    } else {
      NNDEPLOY_LOGE("Decoder not initialized");
      return base::kStatusCodeErrorNotSupport;
    }
  }

 private:
  // FFmpeg GPU解码主循环
  base::Status runFFmpegDecode() {
    while (true) {
      int ret = av_read_frame(fmt_ctx_, packet_);
      if (ret < 0) {
        if (ret == AVERROR_EOF) {
          NNDEPLOY_LOGI("End of stream");
        } else {
          char errbuf[AV_ERROR_MAX_STRING_SIZE];
          av_strerror(ret, errbuf, sizeof(errbuf));
          NNDEPLOY_LOGE("Error reading frame: %s", errbuf);
        }
        return base::kStatusCodeErrorInvalidValue;
      }

      if (packet_->stream_index != video_stream_idx_) {
        av_packet_unref(packet_);
        continue;  // 跳过音频等其他流
      }

      // 发送packet给解码器
      ret = avcodec_send_packet(codec_ctx_, packet_);
      av_packet_unref(packet_);

      if (ret < 0) {
        NNDEPLOY_LOGE("Error sending packet to decoder");
        continue;
      }

      // 接收解码帧
      ret = avcodec_receive_frame(codec_ctx_, frame_);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        continue;  // 需要更多数据
      } else if (ret < 0) {
        NNDEPLOY_LOGE("Error receiving frame from decoder");
        return base::kStatusCodeErrorInvalidValue;
      }

      // GPU帧 → CPU帧（NV12格式）
      if (frame_->format == AV_PIX_FMT_CUDA) {
        ret = av_hwframe_transfer_data(sw_frame_, frame_, 0);
        if (ret < 0) {
          NNDEPLOY_LOGE("Error transferring frame from GPU to CPU");
          av_frame_unref(frame_);
          return base::kStatusCodeErrorInvalidValue;
        }
        av_frame_unref(frame_);
      } else {
        av_frame_move_ref(sw_frame_, frame_);
      }

      // 转换为BGR格式（OpenCV兼容）
      cv::Mat *output_mat = new cv::Mat(height_, width_, CV_8UC3);
      
      if (!sws_ctx_) {
        sws_ctx_ = sws_getContext(
            sw_frame_->width, sw_frame_->height, (AVPixelFormat)sw_frame_->format,
            width_, height_, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
      }

      uint8_t *dst_data[1] = {output_mat->data};
      int dst_linesize[1] = {(int)output_mat->step[0]};
      sws_scale(sws_ctx_, sw_frame_->data, sw_frame_->linesize, 0, sw_frame_->height,
                dst_data, dst_linesize);

      av_frame_unref(sw_frame_);

      outputs_[0]->set(output_mat, false);
      index_++;
      return base::kStatusCodeOk;
    }
  }

  // OpenCV CPU解码（回退）
  base::Status runOpenCVDecode() {
    cv::Mat *frame = new cv::Mat();
    if (!cap_->read(*frame) || frame->empty()) {
      delete frame;
      return base::kStatusCodeErrorInvalidValue;
    }

    outputs_[0]->set(frame, false);
    index_++;
    return base::kStatusCodeOk;
  }

  // FFmpeg成员变量
  AVFormatContext *fmt_ctx_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVBufferRef *hw_device_ctx_ = nullptr;
  AVFrame *frame_ = nullptr;
  AVFrame *sw_frame_ = nullptr;
  AVPacket *packet_ = nullptr;
  SwsContext *sws_ctx_ = nullptr;
  int video_stream_idx_ = -1;

  // OpenCV回退方案
  cv::VideoCapture *cap_ = nullptr;

  // 通用变量
  std::string path_;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 0;
  int index_ = 0;
  bool use_gpu_ = true;
};

/**
 * @brief 可视化检测结果节点 - 在图像上画框并标注
 */
class VisDetection : public dag::Node {
 public:
  VisDetection(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::demo::VisDetection";
    desc_ = "Visualize detection results with bounding boxes and labels";
  }

  virtual ~VisDetection() {}

  virtual base::Status run() {
    // 获取输入：原始图像和检测结果
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    detect::DetectResult *detect_result =
        (detect::DetectResult *)(inputs_[1]->getParam(this));

    if (input_mat == nullptr || detect_result == nullptr) {
      NNDEPLOY_LOGE("Input is nullptr");
      return base::kStatusCodeErrorNullParam;
    }

    // 创建输出图像（使用new分配，Edge会管理生命周期）
    cv::Mat *output_mat = new cv::Mat();
    input_mat->copyTo(*output_mat);

    // 获取图像尺寸（用于坐标缩放）
    int img_w = input_mat->cols;
    int img_h = input_mat->rows;

    // COCO数据集的类别名称（80类）
    static const std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"};

    // 颜色映射（为不同类别生成不同颜色）
    auto getColor = [](int class_id) -> cv::Scalar {
      int offset = class_id * 123457 % 255;
      return cv::Scalar((offset) % 255, (offset + 85) % 255,
                        (offset + 170) % 255);
    };

    // 绘制每个检测框
    for (size_t i = 0; i < detect_result->bboxs_.size(); ++i) {
      const auto &bbox_result = detect_result->bboxs_[i];
      const auto &bbox = bbox_result.bbox_;
      int class_id = bbox_result.label_id_;
      float score = bbox_result.score_;

      // 调试信息：打印检测框坐标
      static int debug_count = 0;
      if (debug_count++ % 30 == 0) {
        NNDEPLOY_LOGI("Drawing bbox[%zu]: class=%d, score=%.2f, coords=[%.1f,%.1f,%.1f,%.1f]",
                      i, class_id, score, bbox[0], bbox[1], bbox[2], bbox[3]);
      }

      // 将归一化坐标转换为像素坐标（YOLO输出是0-1的归一化坐标）
      float x1 = bbox[0] * img_w;
      float y1 = bbox[1] * img_h;
      float x2 = bbox[2] * img_w;
      float y2 = bbox[3] * img_h;

      // 绘制矩形框
      cv::Scalar color = getColor(class_id);
      cv::rectangle(*output_mat, cv::Point(x1, y1),
                    cv::Point(x2, y2), color, 2);

      // 准备标签文本
      std::string label = class_names[class_id] + ": " +
                          std::to_string(score).substr(0, 4);

      // 绘制标签背景
      int baseline;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::rectangle(*output_mat, cv::Point(x1, y1 - label_size.height - 5),
                    cv::Point(x1 + label_size.width, y1), color, -1);

      // 绘制标签文本
      cv::putText(*output_mat, label, cv::Point(x1, y1 - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // 添加FPS显示
    static auto last_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    double fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(
                              current_time - last_time)
                              .count();
    last_time = current_time;

    std::string fps_text = "FPS: " + std::to_string((int)fps);
    cv::putText(*output_mat, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // 实时显示（根据全局标志控制）
    if (g_enable_display) {
      cv::imshow("YOLO Detection", *output_mat);
      // waitKey延迟控制帧率，降低CPU占用
      // 1ms = ~1000 FPS (CPU全速), 30ms = ~33 FPS (流畅), 16ms = ~60 FPS
      int key = cv::waitKey(16);  // 30ms延迟，降低CPU占用
      // 按 'q' 或 ESC 退出
      if (key == 'q' || key == 27) {
        NNDEPLOY_LOGI("User requested exit (pressed 'q' or ESC)");
        // 注意：这里无法直接停止graph，只能记录状态
      }
    }

    // 输出结果（用于保存到文件）
    outputs_[0]->set(output_mat, false);

    return base::kStatusCodeOk;
  }
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 解析命令行参数
  std::string name = demo::getName();
  NNDEPLOY_LOGI("Name: %s", name.c_str());

  base::InferenceType inference_type = demo::getInferenceType();
  NNDEPLOY_LOGI("Got inference type");

  base::DeviceType device_type;
  try {
    device_type = demo::getDeviceType();
    NNDEPLOY_LOGI("Got device type: %s", base::deviceTypeToString(device_type).c_str());
  } catch (...) {
    NNDEPLOY_LOGE("Failed to get device type, using default CPU");
    device_type = base::DeviceType();
    device_type.code_ = base::kDeviceTypeCodeCpu;
    device_type.device_id_ = 0;
  }

  base::ModelType model_type = demo::getModelType();
  NNDEPLOY_LOGI("Got model type");

  bool is_path = demo::isPath();
  std::vector<std::string> model_value = demo::getModelValue();
  NNDEPLOY_LOGI("Model path: %s", model_value.empty() ? "none" : model_value[0].c_str());

  std::string input_path = demo::getInputPath();    // RTSP URL
  std::string output_path = demo::getOutputPath();  // 输出文件路径
  base::CodecFlag codec_flag = demo::getCodecFlag();
  base::ParallelType pt = demo::getParallelType();

  // 检查是否只显示不保存（output_path为空或"-"）
  bool save_to_file = !output_path.empty() && output_path != "-";

  // 如果output_path是"-"，表示只显示不保存
  if (output_path == "-") {
    g_enable_display = true;
    NNDEPLOY_LOGI("Display mode: Real-time window only (no file output)");
  } else if (output_path.empty()) {
    // 如果没有指定output_path，默认只显示
    g_enable_display = true;
    save_to_file = false;
    NNDEPLOY_LOGI("Display mode: Real-time window only (no output path specified)");
  } else {
    // 既显示又保存
    g_enable_display = true;
    NNDEPLOY_LOGI("Display mode: Real-time window + save to file");
  }

  NNDEPLOY_LOGI("===== RTSP + YOLO Real-time Detection =====");
  NNDEPLOY_LOGI("Input (RTSP): %s", input_path.c_str());
  if (save_to_file) {
    NNDEPLOY_LOGI("Output: %s", output_path.c_str());
  } else {
    NNDEPLOY_LOGI("Output: Display only (no file)");
  }
  NNDEPLOY_LOGI("Model: %s", model_value[0].c_str());
  NNDEPLOY_LOGI("Device: %s", base::deviceTypeToString(device_type).c_str());
  NNDEPLOY_LOGI("Inference: %s",
                base::inferenceTypeToString(inference_type).c_str());

  // 创建DAG边
  NNDEPLOY_LOGI("Creating DAG edges...");
  dag::Edge *input = nullptr;
  dag::Edge *detect_result = nullptr;
  dag::Edge *vis_output = nullptr;

  try {
    input = new dag::Edge("decode_out");
    NNDEPLOY_LOGI("Created input edge: %p", input);

    detect_result = new dag::Edge("detect_out");
    NNDEPLOY_LOGI("Created detect_result edge: %p", detect_result);

    vis_output = new dag::Edge("vis_out");
    NNDEPLOY_LOGI("Created vis_output edge: %p", vis_output);
  } catch (...) {
    NNDEPLOY_LOGE("Failed to create edges");
    return -1;
  }

  // 创建Graph
  NNDEPLOY_LOGI("Creating graph...");
  dag::Graph *graph = new dag::Graph("rtsp_yolo_demo", {}, {});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("Failed to create graph");
    return -1;
  }

  // 1. 视频解码节点（支持RTSP over TCP + GPU硬解）
  NNDEPLOY_LOGI("Creating decode node with GPU hardware decode...");
  codec::Decode *decode_node = nullptr;
  try {
    // 使用自定义的RTSP解码器，启用GPU硬件解码
    decode_node = new RtspVideoDecode("decode_node", input, true);  // true = 启用GPU解码
    NNDEPLOY_LOGI("Decode node created: %p", decode_node);
  } catch (const std::exception& e) {
    NNDEPLOY_LOGE("Exception creating decode node: %s", e.what());
    return -1;
  } catch (...) {
    NNDEPLOY_LOGE("Unknown exception creating decode node");
    return -1;
  }

  if (decode_node == nullptr) {
    NNDEPLOY_LOGE("Failed to create decode node - returned nullptr");
    return -1;
  }

  NNDEPLOY_LOGI("Adding decode node to graph...");
  graph->addNode(decode_node);
  NNDEPLOY_LOGI("Decode node added successfully");

  // 2. YOLO检测Graph
  NNDEPLOY_LOGI("Creating YOLO graph...");
  detect::YoloGraph *yolo_graph =
      new detect::YoloGraph(name, {input}, {detect_result});
  if (yolo_graph == nullptr) {
    NNDEPLOY_LOGE("Failed to create YOLO graph");
    return -1;
  }

  // Define node descriptors for YoloGraph internal structure
  std::vector<std::string> model_inputs = {"yolo_in"};
  std::vector<std::string> model_outputs = {"yolo_out"};
  dag::NodeDesc pre_desc("preprocess", {"decode_out"}, model_inputs);
  dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
  dag::NodeDesc post_desc("postprocess", model_outputs, {"detect_out"});

  // Initialize YoloGraph internal nodes
  yolo_graph->make(pre_desc, infer_desc, inference_type, post_desc);
  yolo_graph->setInferParam(device_type, model_type, is_path, model_value);

  // 设置YOLO版本为8（针对YOLOv8模型）
  yolo_graph->setVersion(8);

  graph->addNode(yolo_graph);

  // 3. 可视化节点
  NNDEPLOY_LOGI("Creating visualization node...");
  dag::Node *vis_node =
      graph->createNode<VisDetection>("vis_node", {input, detect_result},
                                      {vis_output});
  if (vis_node == nullptr) {
    NNDEPLOY_LOGE("Failed to create visualization node");
    return -1;
  }

  // 4. 视频编码节点（只在需要保存文件时创建）
  codec::Encode *encode_node = nullptr;
  if (save_to_file) {
    NNDEPLOY_LOGI("Creating encode node...");
    encode_node = codec::createEncode(
        base::kCodecTypeOpenCV, codec_flag, "encode_node", vis_output);
    if (encode_node == nullptr) {
      NNDEPLOY_LOGE("Failed to create encode node");
      return -1;
    }
    graph->addNode(encode_node);
  } else {
    NNDEPLOY_LOGI("Skipping encode node (display only mode)");
  }

  // 设置并行模式
  base::Status status = graph->setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Failed to set parallel type");
    return -1;
  }

  graph->setTimeProfileFlag(true);

  // 初始化Graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Failed to initialize graph");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  // 打印Graph结构
  status = graph->dump();

  // 设置输入输出路径
  decode_node->setPath(input_path);  // RTSP URL: rtsp://...
  if (save_to_file && encode_node != nullptr) {
    encode_node->setRefPath(input_path);
    encode_node->setPath(output_path);
  }

  // 运行推理循环
  NNDEPLOY_TIME_POINT_START("graph->run");
  int frame_count = 0;
  int max_frames = decode_node->getSize();  // 如果是RTSP流，这可能是-1

  NNDEPLOY_LOGI("Starting inference loop...");
  NNDEPLOY_LOGI("Press Ctrl+C to stop");

  // 对于RTSP流，循环直到流结束或用户中断
  while (true) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGI("Stream ended or error occurred, exiting...");
      break;
    }

    frame_count++;

    // 如果不是pipeline模式，可以访问结果
    if (pt != base::kParallelTypePipeline) {
      detect::DetectResult *result =
          (detect::DetectResult *)detect_result->getGraphOutputParam();
      if (result != nullptr) {
        // 可选：打印检测统计
        if (frame_count % 30 == 0) {  // 每30帧打印一次
          NNDEPLOY_LOGI("Frame %d: Detected %zu objects", frame_count,
                        result->bboxs_.size());
        }
      }
    }

    // 如果指定了最大帧数（用于测试）
    if (max_frames > 0 && frame_count >= max_frames) {
      NNDEPLOY_LOGI("Reached max frames: %d", max_frames);
      break;
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");

  NNDEPLOY_LOGI("Processed %d frames", frame_count);

  // 反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Failed to deinitialize graph");
    return -1;
  }

  // 打印性能统计
  NNDEPLOY_TIME_PROFILER_PRINT("rtsp_yolo_demo");

  // 清理资源
  delete input;
  delete detect_result;
  delete vis_output;
  if (encode_node != nullptr) {
    delete encode_node;
  }
  delete decode_node;
  delete yolo_graph;
  delete graph;

  // 关闭OpenCV窗口
  if (g_enable_display) {
    cv::destroyAllWindows();
  }

  NNDEPLOY_LOGI("Demo finished successfully!");

  return 0;
}
