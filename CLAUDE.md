# nndeploy Codebase Architecture Guide

## Project Overview

**nndeploy** is an easy-to-use and high-performance AI deployment framework designed for multi-platform inference. It provides:
- Visual workflow editor for AI model deployment
- Unified multi-backend inference support (13+ frameworks)
- Python and C++ APIs
- Plugin architecture for custom nodes
- Multi-level parallelization (task-level, pipeline-level)
- Memory optimization and resource pooling
- Cross-platform support (Linux, Windows, macOS, Android, iOS)

**Current Version**: 3.0.7  
**Language**: C++17 (primary), Python 3.10+  
**Build System**: CMake 3.12+

---

## Build System & Development Workflow

### Build Commands

#### Quick Setup

```bash
# 1. Clone repository with submodules
git clone https://github.com/nndeploy/nndeploy.git
cd nndeploy
git submodule update --init --recursive
# Or if submodule fetch fails:
python3 clone_submodule.py

# 2. Create build directory with configuration
mkdir build
cp cmake/config.cmake build/
cd build
vim config.cmake  # Edit configuration as needed

# 3. Build
cmake ..
make -j$(nproc)  # Parallel build using all cores
make install

# 4. Verify installation
python -c "import nndeploy; print(nndeploy.__version__)"
```

#### Platform-Specific Build Scripts

The repository provides automated build scripts for different platforms:

- **Linux**: `build_linux.py --config config_opencv_ort_mnn_tokenizer.cmake --build-type Release --jobs 8`
- **macOS ARM64**: `build_mac_arm64.py`
- **Windows**: `build_win.py`

```bash
# Example Linux build with full configuration
python3 build_linux.py \
  --config config_opencv_ort_mnn_tokenizer.cmake \
  --build-type Release \
  --jobs 8 \
  --clean
```

### Configuration Categories

Configuration options in `cmake/config.cmake`:

1. **Basic Build Options** (recommended defaults):
   - `ENABLE_NNDEPLOY_BUILD_SHARED`: Build as shared library (default: ON)
   - `ENABLE_NNDEPLOY_CXX17_ABI`: C++17 standard (default: ON)
   - `ENABLE_NNDEPLOY_TIME_PROFILER`: Enable timing analysis (default: ON)

2. **Core Module Options**:
   - `ENABLE_NNDEPLOY_BASE`: Fundamental utilities (default: ON)
   - `ENABLE_NNDEPLOY_DEVICE`: Device abstraction (default: ON)
   - `ENABLE_NNDEPLOY_INFERENCE`: Inference engines (default: ON)
   - `ENABLE_NNDEPLOY_DAG`: Directed acyclic graph (default: ON)
   - `ENABLE_NNDEPLOY_PYTHON`: Python bindings (default: ON)

3. **Device Backend Options** (all optional, OFF by default):
   - `ENABLE_NNDEPLOY_DEVICE_CUDA`: NVIDIA CUDA
   - `ENABLE_NNDEPLOY_DEVICE_OPENCL`: OpenCL acceleration
   - `ENABLE_NNDEPLOY_DEVICE_METAL`: Apple Metal
   - `ENABLE_NNDEPLOY_DEVICE_ASCEND_CL`: Huawei Ascend

4. **Inference Backend Options** (all optional):
   - `ENABLE_NNDEPLOY_INFERENCE_TENSORRT`: NVIDIA TensorRT
   - `ENABLE_NNDEPLOY_INFERENCE_OPENVINO`: Intel OpenVINO
   - `ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME`: ONNX Runtime
   - `ENABLE_NNDEPLOY_INFERENCE_MNN`: Alibaba MNN
   - `ENABLE_NNDEPLOY_INFERENCE_TNN`: Tencent TNN
   - `ENABLE_NNDEPLOY_INFERENCE_NCNN`: NCNN
   - Plus: CoreML, TVM, PaddleLite, RKNN, Ascend CL, SNPE, TensorFlow, PyTorch

5. **Algorithm Plugins** (CV algorithms ON by default):
   - Requires `ENABLE_NNDEPLOY_OPENCV` for detection, segmentation, classification
   - Language/AIGC models require `ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP` (depends on Rust)

### Build Outputs

After successful compilation:
- **Main library**: `libnndeploy_framework.so` (or `.dll` on Windows)
- **Algorithm plugins**: `libnndeploy_plugin_<type>.so`
- **Demo executables**: `nndeploy_demo_<name>`
- **Python package**: Built and installable via `pip install -e .`

### Python Installation

```bash
# Option 1: PyPI (recommended for users)
pip install --upgrade nndeploy

# Option 2: Development mode (for developers)
cd python
pip install -e .

# Troubleshooting
# Virtual environment recommended:
python3 -m venv nndeploy_env
source nndeploy_env/bin/activate
pip install -e .

# If dynamic library path issues:
export LD_LIBRARY_PATH=/path/to/nndeploy/python/nndeploy:$LD_LIBRARY_PATH
```

---

## High-Level Architecture Overview

### Architectural Layers (Bottom-to-Top)

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│         (Visual Workflow, Python/C++ API, CLI Tools)         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────────┐
│                Plugin & Algorithm Layer                      │
│  (Custom Nodes, Inference Nodes, Pre/Post-processing)       │
│  Located: plugin/source/nndeploy/                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────────┐
│         DAG (Directed Acyclic Graph) & Execution Layer      │
│  (Graph, Nodes, Edges, Executors, Parallel Scheduling)     │
│  Located: framework/include/nndeploy/dag/                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────────┐
│              Inference Abstraction Layer                     │
│  (13+ Backend Wrappers: TensorRT, OpenVINO, MNN, etc.)      │
│  Located: framework/source/nndeploy/inference/              │
│  Process Templates: Infer nodes abstracting model details    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────────┐
│         Device Management & Kernel Layer                     │
│  (CPU, GPU, NPU abstraction, Memory, Tensors, Operators)    │
│  Located: framework/source/nndeploy/device/                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────────┐
│                   Base Infrastructure                        │
│  (Status, Common Types, Logging, Thread Pool, Memory Pool)   │
│  Located: framework/source/nndeploy/base/                   │
└─────────────────────────────────────────────────────────────┘
```

### Core Architectural Principles

1. **Abstraction Over Implementation**: All hardware/inference backends are abstracted behind unified interfaces
2. **Plugin Architecture**: New algorithms and operations are plugins, not core dependencies
3. **Graph-Based Execution**: All computation expressed as DAGs for maximum parallelization opportunities
4. **Zero-Copy Data Flow**: Memory is managed at device level with reference counting
5. **Cross-Language Support**: Core C++ with Python bindings via pybind11

---

## Key Components & Their Interactions

### 1. **BASE Module** (`framework/source/nndeploy/base/`)

Core infrastructure shared across all modules:

**Key Files**:
- `common.h/cc`: Enums for device types, inference backends, model formats, status codes
- `status.h`: Status/error handling throughout the framework
- `object.h`: Base class for all framework objects (reference counting, polymorphism)
- `param.h`: Base class for all parameter structures
- `log.h`: Logging utilities with profiling support
- `macro.h`: Compiler macros for API visibility, type checking

**Responsibility**: Provide unified type definitions and error handling so all modules speak the same language.

### 2. **DEVICE Module** (`framework/source/nndeploy/device/`)

Abstracts hardware and memory management across different devices.

**Architecture Pattern**:
```cpp
Architecture (abstract) ─┬─> CPUArchitecture ──> CPUDevice
                        ├─> CudaArchitecture ──> CudaDevice
                        ├─> ArmArchitecture ──> ArmDevice
                        └─> [16+ other devices]
```

**Key Components**:

- **Device & Architecture Classes**:
  - `Architecture`: Factory pattern for creating device instances
  - `Device`: Unified interface for memory operations, synchronization, device info
  - Devices: CPU, ARM, X86, CUDA, OpenCL, Metal, Ascend CL, etc.

- **Data Containers**:
  - `Buffer`: Raw memory abstraction with device type tracking
  - `Tensor`: N-dimensional array with shape, dtype, device info
  - `Mat`: 2D image container (width, height, channels, pixel type)
  - `TensorDesc`: Metadata describing tensor shape, datatype, memory layout
  - `BufferDesc`: Metadata for buffer allocation

- **Memory Management**:
  - `MemoryPool`: Pre-allocates and recycles memory to avoid allocation overhead
  - Device-specific implementations (CUDA uses cudaMallocAsync, etc.)

- **Stream & Synchronization**:
  - `Stream`: Command queue for device operations (GPU streams, etc.)
  - `Event`: Synchronization primitive across streams

**Key Files**:
- `device.h`: Architecture/Device base classes
- `tensor.h`: Tensor class with creation/manipulation methods
- `buffer.h`: Buffer memory abstraction
- `type.h`: Tensor/Buffer descriptor structures
- Subdirs: `cpu/`, `cuda/`, `arm/`, `ascend_cl/`, etc.

**Data Flow Example**:
```
User Code
    ↓
Device *device = getDevice(kDeviceTypeCudaFloat32)
    ↓
Tensor *tensor = new Tensor(device, tensor_desc)  [Memory allocated by device]
    ↓
Buffer *buf = device->allocate(size)  [Memory managed by device's memory pool]
    ↓
device->copy(src, dst)  [Cross-device memory transfer handled transparently]
```

### 3. **INFERENCE Module** (`framework/source/nndeploy/inference/`)

Unified interface to 13+ inference frameworks.

**Architecture Pattern** (Factory + Bridge):
```
InferenceParam (base)
    ├─> TensorRTParam
    ├─> OnnxRuntimeParam
    ├─> MnnParam
    └─> [13+ backend params]

Inference (base, abstract interface)
    ├─> TensorRTInference ─┬─> TensorRTConverter
    ├─> OnnxRuntimeInference ─┬─> OnnxRuntimeConverter
    ├─> MnnInference ─┬─> MnnConverter
    └─> [13+ implementations]
```

**Unified Inference Interface**:
```cpp
Inference* inference = createInference(param);
status = inference->init();
// Set inputs
inference->reshape(shape_map);  // For dynamic shapes
// Run inference
status = inference->forward();
// Get outputs
```

**Supported Backends**:
1. ONNXRuntime (ONNX models)
2. TensorRT (NVIDIA optimization)
3. OpenVINO (Intel optimization)
4. MNN (Alibaba, mobile-optimized)
5. TNN (Tencent)
6. NCNN (light-weight)
7. CoreML (Apple)
8. TVM (Apache)
9. PaddleLite (Baidu)
10. RKNN (Rockchip)
11. AscendCL (Huawei)
12. SNPE (Qualcomm)
13. PyTorch/TensorFlow
14. Custom/Default backend (built-in)

**Key Files**:
- `inference.h`: Base Inference class, unified interface
- `inference_param.h`: InferenceParam base class
- Framework dirs: `onnxruntime/`, `tensorrt/`, `mnn/`, `openvino/`, etc.
- Each framework has: `xxx_inference.h/cc`, `xxx_inference_param.h/cc`, `xxx_converter.h/cc`

**Data Flow**:
```
Model File + InferenceParam
    ↓
Inference::init() [Loads model, prepares accelerators]
    ↓
Set input tensors via framework-specific converter
    ↓
Inference::forward() [Executes on device]
    ↓
Get output tensors via framework-specific converter
```

### 4. **DAG Module** (`framework/source/nndeploy/dag/`)

Directed acyclic graph execution engine - the heart of nndeploy.

**Core Concepts**:

- **Node**: Computational unit (base class in `node.h`)
  - Input/Output edges connected to other nodes
  - `run()` method contains computation logic
  - Parameters for configuration
  - Inheritable for custom operations (inference, pre-processing, etc.)

- **Edge**: Data channel between nodes
  - Carries Tensor, Mat, or custom data objects
  - Two variants: `Edge` (single value) and `PipelineEdge` (multi-queue for pipelining)
  - Automatic data type checking

- **Graph**: Container organizing nodes and edges
  - Builds topological ordering
  - Tracks node dependencies
  - Serializable to JSON for visual workflows

**Built-in Node Types**:
- **Inference/Infer**: Wraps Inference backend with automatic tensor conversion
- **Composite/SubGraph**: Hierarchical graphs (graphs within graphs)
- **ConstNode**: Provides constant data
- **Loop**: Iterative execution
- **Condition**: Conditional branching (if/else)

**Key Files**:
- `node.h`: Base Node class (43KB, extensive templating)
- `edge.h`: Edge/PipelineEdge classes (18KB)
- `graph.h`: Graph management (88KB, contains serialization logic)
- `executor/`: Execution engines
  - `parallel_task_executor.h`: Task-level parallelism
  - `parallel_pipeline_executor.h`: Pipeline parallelism
  - `executor.h`: Base executor interface
- `util.h`: Graph utility functions (topology sort, serialization)

**Graph Execution Flow**:
```
Load/Create Graph
    ↓
graph->init()  [Initialize all nodes and edges]
    ↓
Select Executor (parallel_task_executor, parallel_pipeline_executor, etc.)
    ↓
executor->run(graph)
    ├─ Topological sort nodes
    ├─ Schedule on thread pool
    ├─ Respect edge dependencies
    └─ Collect results
    ↓
graph->deinit()
```

### 5. **PLUGIN Module** (`plugin/source/nndeploy/`)

High-level algorithm implementations using DAG primitives.

**Plugin Categories**:

1. **infer**: Process templates and utility nodes
   - `Infer<T>`: Template class wrapping `Inference` with auto tensor conversion
   - Input tensor → model-specific preprocessing → inference → output tensor

2. **basic**: Pre/post-processing operations
   - Image resizing, normalization, color conversion
   - Uses OpenCV or custom optimized implementations

3. **detect**: Object detection algorithms
   - YOLOv5/v6/v7/v8/v11
   - Node definitions + post-processing

4. **segment**: Image segmentation
   - Segment Anything, RMBG, etc.
   - Mask extraction and visualization nodes

5. **classification**: Image classification
   - ResNet, MobileNet, EfficientNet, etc.

6. **llm**: Large language models
   - Tokenizer integration
   - Token sampling nodes
   - Embedding and KV-cache management

7. **tokenizer**: Text processing
   - `TokenizerText`: Wraps tokenizer libraries
   - `TokenizerEncodeCpp`: CPP-based tokenization
   - `TokenizerMnn`: MNN-based tokenization

8. **gan**: Generative models
   - Face swap, style transfer, etc.

9. **stable_diffusion**: Text-to-image generation
   - Embedding, VAE, UNet, scheduling

**Plugin Registration**:
```cpp
// In plugin implementation
REGISTER_NODE("nndeploy::example::MyNode", MyNode);

// User loads in frontend:
nndeploy-app --port 8000 --plugin /path/to/libnndeploy_plugin_custom.so
```

### 6. **THREAD POOL** (`framework/source/nndeploy/thread_pool/`)

Efficient task scheduling for parallel execution.

**Design**:
- Each thread has private task queue + work-stealing capability
- Based on CThreadPool library
- Returns std::future for async-to-sync conversion
- Used by DAG executors to parallelize node execution

**Key Methods**:
```cpp
thread_pool->commit([](){ /* task */})  → std::future<T>
future.get()  // Block until completion
```

### 7. **IR & NET Modules** (Less critical, framework-specific)

- **IR** (`framework/source/nndeploy/ir/`): Intermediate representation handling
- **NET** (`framework/source/nndeploy/net/`): Neural network graph representation (ONNX-like)
- **OP** (`framework/source/nndeploy/op/`): High-performance operators (CPU, CUDA, OpenCL)

---

## Parallel Execution Strategies

### Task-Level Parallelism (ParallelTaskExecutor)

**Use Case**: Multiple independent models or processing paths

**How It Works**:
1. Topological sort nodes
2. Identify nodes with no dependencies (ready to execute)
3. Submit to thread pool
4. After each node finishes, check if its dependent nodes are now ready
5. Recursively submit ready nodes
6. Synchronize when all complete

**Example DAG**:
```
    ┌─→ Model A ──┐
Input ┤            ├─→ Merge ─→ Output
    └─→ Model B ──┘
```

Model A and B execute in parallel; Merge waits for both.

**Code Reference**: `framework/include/nndeploy/dag/executor/parallel_task_executor.h`

### Pipeline Parallelism (ParallelPipelineExecutor)

**Use Case**: Processing multiple frames/batches (video, streaming)

**How It Works**:
```
Time →
t0: Preprocess(Frame0)
t1: Preprocess(Frame1) | Infer(Frame0_preprocessed)
t2: Preprocess(Frame2) | Infer(Frame1_preprocessed) | Postprocess(Frame0_inferred)
t3:                     | Infer(Frame2_preprocessed) | Postprocess(Frame1_inferred)
t4:                                                    | Postprocess(Frame2_inferred)
```

- Each stage (preprocess/infer/postprocess) runs on separate thread
- Stages operate on different data concurrently
- Throughput limited by slowest stage
- Latency for first frame = sum of all stages

**Data Structure**:
- `PipelineEdge`: Maintains queue of data packets + per-consumer indices
- Automatically handles data buffering and cleanup

**Code Reference**: `framework/include/nndeploy/dag/executor/parallel_pipeline_executor.h`

---

## Plugin Architecture & Custom Node Development

### C++ Custom Nodes

**File Structure**:
```
plugin/
├── include/nndeploy/plugin/
│   └── my_custom_node.h
├── source/nndeploy/plugin/
│   └── my_custom_node.cc
└── CMakeLists.txt
```

**Minimal Implementation**:
```cpp
// my_custom_node.h
class MyCustomNode : public dag::Node {
public:
    MyCustomNode(const std::string& name,
                 std::vector<dag::Edge*> inputs,
                 std::vector<dag::Edge*> outputs)
        : dag::Node(name, inputs, outputs) {
        key_ = "nndeploy::example::MyCustomNode";
        param_ = std::make_shared<MyParam>();
        setInputTypeInfo<device::Tensor>();
        setOutputTypeInfo<device::Tensor>();
    }
    
    base::Status run() override {
        device::Tensor* input = (device::Tensor*)(getInput(0)->getTensor(this));
        // Process input
        device::Tensor* output = getOutput(0)->create(device, tensor_desc);
        // Set output
        return base::kStatusCodeOk;
    }
};

// Register
REGISTER_NODE("nndeploy::example::MyCustomNode", MyCustomNode);
```

**Compilation & Loading**:
```bash
# Compile as plugin library
g++ -shared -fPIC my_custom_node.cc -o libnndeploy_plugin_custom.so

# Load in app
nndeploy-app --port 8000 --plugin /path/to/libnndeploy_plugin_custom.so
```

### Python Custom Nodes

**File Structure**:
```
plugin/
├── template.py
└── app.py
```

**Minimal Implementation**:
```python
import nndeploy.dag
import nndeploy.base

class MyCustomNode(nndeploy.dag.Node):
    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.example.MyCustomNode")
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
    def run(self):
        input_data = self.get_input_data(0)
        output_data = process(input_data)
        self.set_output_data(output_data, 0)
        return nndeploy.base.Status.ok()

# Register
class MyCustomNodeCreator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        return MyCustomNode(name, inputs, outputs)

nndeploy.dag.register_node("nndeploy.example.MyCustomNode", MyCustomNodeCreator())
```

**Loading**:
```bash
nndeploy-app --port 8000 --plugin /path/to/template.py
```

---

## Adding New Inference Backends

**5-Step Process**:

### 1. Add Enumerations

```cpp
// include/nndeploy/base/common.h
enum InferenceType {
    kInferenceTypeXxx,  // Add new backend
    // ...
};

// source/nndeploy/base/common.cc
InferenceType stringToInferenceType(const std::string& src) {
    if (src == "Xxx") return kInferenceTypeXxx;
    // ...
}
```

### 2. Define InferenceParam

```cpp
// include/nndeploy/inference/xxx/xxx_inference_param.h
class XxxInferenceParam : public InferenceParam {
public:
    // Framework-specific options
    std::string some_option;
};
```

### 3. Implement Inference Class

```cpp
// include/nndeploy/inference/xxx/xxx_inference.h
class XxxInference : public Inference {
public:
    XxxInference(base::InferenceType type);
    
    base::Status init() override;
    base::Status deinit() override;
    base::Status reshape(base::ShapeMap& shape_map) override;
    // ... inference framework-specific implementations
};
```

### 4. Implement Converter

```cpp
// include/nndeploy/inference/xxx/xxx_converter.h
class XxxConverter {
public:
    static base::Status toFrameworkTensor(device::Tensor* src, void** dst);
    static base::Status fromFrameworkTensor(void* src, device::Tensor* dst);
};
```

Handles conversion between nndeploy's unified Tensor format and framework-specific tensor types.

### 5. Update CMake

```cmake
# CMakeLists.txt
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_XXX "Enable Xxx backend" OFF)

if(ENABLE_NNDEPLOY_INFERENCE_XXX)
    file(GLOB_RECURSE INFERENCE_XXX_SOURCE
        "${ROOT_PATH}/include/nndeploy/inference/xxx/*.h"
        "${ROOT_PATH}/source/nndeploy/inference/xxx/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_XXX_SOURCE})
endif()

# cmake/xxx.cmake - Create library linking file
include("${ROOT_PATH}/cmake/xxx.cmake")  # in nndeploy.cmake
```

---

## Python Bindings Architecture

### Binding Structure

Python modules in `python/src/` mirror C++ structure:
```
python/src/
├── base/           → C++ base module (status, common types)
├── device/         → C++ device module (Tensor, Buffer, Device)
├── inference/      → C++ inference module (Inference backends)
├── dag/            → C++ DAG module (Graph, Node, Edge)
├── infer/          → C++ infer template wrappers
├── [algorithms]/   → C++ plugin algorithms (detect, segment, etc.)
└── server/         → Python-only web server for visual editor
```

### Binding Implementation

Uses **pybind11** for C++-Python interoperability:

```cpp
// python/src/base/base.cc
PYBIND11_MODULE(nndeploy_base, m) {
    py::class_<base::Status>(m, "Status")
        .def(py::init<>())
        .def("code", &base::Status::getStatusCode)
        .def("desc", &base::Status::getDesc);
    
    py::class_<device::Tensor>(m, "Tensor")
        .def(py::init<Device*, TensorDesc>())
        .def("shape", &device::Tensor::getShape)
        .def("data", &device::Tensor::getData);
}
```

### Python API Entry Points

```python
import nndeploy

# High-level APIs
nndeploy.dag.Graph                 # Graph management
nndeploy.dag.Node                  # Base node class
nndeploy.device.Tensor             # Tensor operations
nndeploy.inference.Inference        # Inference interface
nndeploy.base.Status                # Error handling

# Algorithm plugins
nndeploy.detect.YOLOv8             # Detection
nndeploy.segment.SegmentAnything   # Segmentation
nndeploy.llm.Qwen                  # Language models
nndeploy.stable_diffusion.Diffusion # Image generation
```

---

## Visual Workflow System

### Frontend-Backend Communication

1. **Visual Editor** (`app/workflow/`): Vue.js-based node editor
2. **Backend API** (`python/nndeploy/server/`): Flask/Uvicorn web server
3. **JSON Serialization** (`framework/include/nndeploy/dag/graph.h`): Serialize/deserialize graphs

### Workflow JSON Format

```json
{
  "nodes": [
    {
      "id": "input_0",
      "name": "input",
      "type": "nndeploy::base::Input",
      "param": { "shape": [1, 3, 640, 640] }
    },
    {
      "id": "infer_yolo",
      "name": "YOLOv8 Detection",
      "type": "nndeploy::infer::Infer",
      "param": {
        "model_path": "yolov8s.onnx",
        "inference_type": "onnxruntime",
        "device_type": "cpu"
      }
    },
    {
      "id": "output_0",
      "name": "output",
      "type": "nndeploy::base::Output"
    }
  ],
  "edges": [
    { "src": "input_0", "dst": "infer_yolo", "src_edge": 0, "dst_edge": 0 },
    { "src": "infer_yolo", "dst": "output_0", "src_edge": 0, "dst_edge": 0 }
  ]
}
```

### Workflow Execution in Code

```python
# Python
graph = nndeploy.dag.Graph("")
graph.load_file("workflow.json")
graph.init()

input_edge = graph.get_input(0)
input_data = nndeploy.device.Tensor(...)
input_edge.set(input_data)

status = graph.run()

output_edge = graph.get_output(0)
result = output_edge.get()
graph.deinit()
```

```cpp
// C++
auto graph = std::make_shared<dag::Graph>("");
graph->loadFile("workflow.json");
graph->removeInOutNode();
graph->init();

dag::Edge* input = graph->getInput(0);
input->set(input_data, false);

graph->run();

dag::Edge* output = graph->getOutput(0);
auto result = output->getGraphOutput<device::Tensor>();
graph->deinit();
```

---

## Memory Management & Optimization

### Memory Hierarchy

```
┌─ User Application
│      ↓
├─ nndeploy DAG Graph
│      ↓
├─ Node (allocates tensors)
│      ↓
├─ Edge (passes tensor references)
│      ↓
├─ Device/MemoryPool (manages physical memory)
│      ↓
├─ Hardware (GPU VRAM, CPU RAM, etc.)
```

### Zero-Copy Architecture

- **Input Sharing**: Multiple nodes can read same input tensor (reference counted)
- **Output Reuse**: Memory can be recycled between inference iterations
- **In-Place Operations**: Some ops modify tensors in-place

### Memory Pool

- Pre-allocates chunks matching common tensor sizes
- Reduces allocation latency (critical for real-time)
- Tracks peak memory usage (profiling)
- Device-specific implementations (CUDA, OpenCL, etc.)

---

## Testing Infrastructure

### Test Organization

```
framework/
├── test/          # Unit tests for each module
│   ├── base/
│   ├── device/
│   ├── inference/
│   ├── dag/
│   └── ...
demo/               # Example applications
├── base/
├── detect/
├── segment/
├── llm/
├── stable_diffusion/
└── ...
```

### Running Tests

```bash
mkdir build && cd build
cmake .. -DENABLE_NNDEPLOY_TEST=ON
make -j
./nndeploy_test_<module>
```

---

## Development Workflow Best Practices

### 1. Building for Development

```bash
mkdir cmake-build-debug
cd cmake-build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_NNDEPLOY_TEST=ON
make -j$(nproc)
```

### 2. Adding a Feature

1. **Modify CMakeLists.txt**: Add source files
2. **Implement in C++**: Core logic
3. **Export Python bindings** (if user-facing): `python/src/<module>/`
4. **Add unit tests**: `framework/test/<module>/`
5. **Document**: Update README, architecture guides, docstrings
6. **Test on multiple platforms**: Linux (primary), Windows, macOS, Android
7. **Verify with demos**: Run existing demos to ensure no regression

### 3. Debugging

```bash
# Enable profiling
cmake .. -DENABLE_NNDEPLOY_TIME_PROFILER=ON

# Address sanitizer (find memory bugs)
cmake .. -DENABLE_NNDEPLOY_ADDRESS_SANTIZER=ON

# Verbose logging
export NNDEPLOY_LOG_LEVEL=DEBUG

# GDB debugging
gdb ./nndeploy_demo_<name>
```

### 4. Code Style

- **Format**: `.clang-format` configuration (LLVM style)
- **Command**: `clang-format -i <file.cc>`
- **Check before commit**: `make format-check`

---

## Key Design Patterns

### 1. Abstract Factory (Devices & Inference)

```cpp
// Get device abstraction
Architecture* arch = getArchitecture(kDeviceTypeCuda);
Device* device = arch->getDevice(0);

// Get inference backend
Inference* inf = createInference(param);
```

### 2. Strategy Pattern (Executors)

```cpp
Executor* executor;
if (use_pipelining) {
    executor = new ParallelPipelineExecutor();
} else {
    executor = new ParallelTaskExecutor();
}
executor->run(graph);
```

### 3. Template Method (Inference Base Class)

```cpp
// Base class defines flow
class Inference {
    virtual Status init() = 0;
    virtual Status forward() = 0;
    
    // Common utilities derived from base
    Status setParam(Param* p);
};
```

### 4. Visitor Pattern (DAG Traversal)

```cpp
graph->topologicalSort();  // Visits all nodes in dependency order
```

### 5. Plugin Registry (Reflection-like)

```cpp
REGISTER_NODE("nndeploy::example::MyNode", MyNode);
auto node = createNode("nndeploy::example::MyNode", ...);
```

---

## Performance Characteristics

### Latency Optimization

- **Task Parallelism**: 7-12% improvement (multi-model, low barrier to parallelize)
- **Pipeline Parallelism**: 13-57% improvement (video/streaming, depends on backend)
- **Memory Pooling**: 20-40% allocation latency reduction
- **Zero-Copy**: Negligible overhead for tensor passing

### Typical End-to-End Latencies

(Benchmarked on RTX3060, i7-12700)

| Task | ONNXRuntime | OpenVINO | TensorRT |
|------|------------|----------|----------|
| YOLOv11s (serial) | 54.8ms | 34.1ms | 13.2ms |
| YOLOv11s (pipeline) | 47.3ms | 29.7ms | 5.7ms |
| Multi-task (serial) | 654.3ms | 489.9ms | 59.1ms |
| Multi-task (task-parallel) | 602.1ms | 435.2ms | 51.9ms |

---

## Common Integration Scenarios

### Scenario 1: Inference Only (No Custom Processing)

```python
import nndeploy

# Minimal setup
inference = nndeploy.inference.Inference(param)
inference.init()

input_tensor = create_tensor_from_image(image)
inference.set_input_tensor(input_tensor, 0)
inference.forward()

output = inference.get_output_tensor(0)
```

### Scenario 2: Graph with Custom Pre/Post-Processing

```python
graph = nndeploy.dag.Graph("detection_pipeline")

# Input node
input_edge = graph.create_edge("input")

# Custom preprocessing
preprocess_node = graph.create_node("custom_preprocess", [input_edge], [...])

# Inference
infer_node = graph.create_infer("yolo", InferenceType.ONNXRUNTIME, [...], [...])

# Custom postprocessing
postprocess_node = graph.create_node("custom_postprocess", [...], [...])

# Output
output_edge = graph.create_edge("output")

graph.init()
graph.run()
```

### Scenario 3: Multi-Model Ensemble

```python
# Build graph with multiple inference branches
input → [Model A, Model B, Model C in parallel] → Merge → Output

# Models execute concurrently, results merged by postprocess node
```

### Scenario 4: Interactive Visual Workflow

```bash
# Start visual editor
python app.py --port 8000

# Build workflow in browser at http://localhost:8000
# Save as workflow.json

# In deployment:
python -c "
import nndeploy
graph = nndeploy.dag.Graph('')
graph.load_file('workflow.json')
graph.init()
# ... set inputs and run ...
graph.deinit()
"
```

---

## Troubleshooting & Common Issues

### Compilation Issues

**Missing CUDA**: If CUDA-based inference is needed:
```bash
# Install CUDA toolkit, then:
cmake .. -DENABLE_NNDEPLOY_DEVICE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

**OpenCV not found**:
```bash
# Either install system package:
sudo apt-get install libopencv-dev

# Or specify path:
cmake .. -DENABLE_NNDEPLOY_OPENCV=/path/to/opencv
```

**Python binding issues**:
```bash
export LD_LIBRARY_PATH=/path/to/nndeploy/build/install/lib:$LD_LIBRARY_PATH
python -c "import nndeploy; print(nndeploy.__version__)"
```

### Runtime Issues

**Model loading fails**:
- Check model path is absolute
- Verify model format matches inference type (ONNX for ONNXRuntime, etc.)
- Check inference backend is enabled in config.cmake

**Out of memory**:
- Reduce batch size
- Enable memory pooling (default ON)
- Monitor with `ENABLE_NNDEPLOY_GPU_MEM_TRACKER=ON`

**Inference mismatch**:
- Verify input shape matches model expectations
- Check data type conversions (uint8 → float32)
- Check normalization/preprocessing matches training

---

## References & Further Reading

### Official Documentation
- **Build Guide**: `docs/zh_cn/quick_start/build.md`
- **Architecture Guide**: `docs/zh_cn/architecture_guide/`
- **Quick Start**: `docs/zh_cn/quick_start/`
- **Developer Guide**: `docs/zh_cn/developer_guide/`
- **API Reference**: https://nndeploy-zh.readthedocs.io/

### Key Design Documents
- `docs/zh_cn/knowledge_shared/nndeploy_from_requirement_analysis_to_architecture_design_v2.md`
- `docs/zh_cn/discussion/graph.md`
- `docs/zh_cn/discussion/python.md`

### External References
- **DAG Inspiration**: CGraph library
- **Thread Pool**: CThreadPool library
- **Inference Frameworks**: TensorRT, OpenVINO, ONNX Runtime, MNN, TNN, NCNN, etc.
- **Visual Editor**: Vue.js-based custom workflow designer

---

## Summary

nndeploy's architecture is built on **abstraction layers** that progressively hide complexity:

1. **Base** provides unified type system
2. **Device** abstracts hardware differences
3. **Inference** unifies 13+ backend APIs
4. **DAG** enables composable, parallelizable pipelines
5. **Plugins** implement domain-specific algorithms
6. **Python** provides accessible high-level API
7. **Visual Editor** makes deployment accessible to non-programmers

The result is a framework where developers can:
- Write once, deploy anywhere (cloud, desktop, mobile, edge)
- Mix Python and C++ components seamlessly
- Leverage parallel execution automatically
- Choose inference backend based on hardware
- Extend with custom nodes/algorithms
- Visualize and debug workflows interactively

