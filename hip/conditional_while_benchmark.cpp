#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr unsigned int kIterations = 5;
constexpr unsigned int kLaunches = 100;

void check(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void conditional_body(unsigned int* counter,
                                 hipGraphConditionalHandle handle,
                                 unsigned int limit) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const unsigned int next = atomicAdd(counter, 1) + 1;
    hipGraphSetConditional(handle, next < limit ? 1u : 0u);
  }
}

__global__ void static_body(unsigned int* counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(counter, 1);
}

template <typename Launch>
double average(Launch launch, unsigned int* counter, const char* label) {
  for (unsigned int i = 0; i < 10; ++i) {
    check(hipMemset(counter, 0, sizeof(*counter)), label);
    launch();
    check(hipDeviceSynchronize(), label);
  }
  const auto begin = std::chrono::steady_clock::now();
  for (unsigned int i = 0; i < kLaunches; ++i) {
    check(hipMemset(counter, 0, sizeof(*counter)), label);
    launch();
    check(hipDeviceSynchronize(), label);
  }
  const auto end = std::chrono::steady_clock::now();
  unsigned int result = 0;
  check(hipMemcpy(&result, counter, sizeof(result), hipMemcpyDeviceToHost), label);
  if (result != kIterations) std::exit(2);
  return std::chrono::duration<double, std::milli>(end - begin).count() / kLaunches;
}

}  // namespace

int main() {
  unsigned int* counter = nullptr;
  check(hipMalloc(&counter, sizeof(*counter)), "hipMalloc(counter)");

  hipGraph_t conditional_outer = nullptr;
  hipGraph_t conditional_body_graph = nullptr;
  check(hipGraphCreate(&conditional_outer, 0), "hipGraphCreate(conditional outer)");
  check(hipGraphCreate(&conditional_body_graph, 0), "hipGraphCreate(conditional body)");

  hipGraphConditionalHandle handle = 0;
  check(hipGraphConditionalHandleCreate(&handle, conditional_outer, 1,
                                        hipGraphCondAssignDefault),
        "hipGraphConditionalHandleCreate");
  unsigned int limit = kIterations;
  void* conditional_args[] = {&counter, &handle, &limit};
  hipKernelNodeParams conditional_kernel{};
  conditional_kernel.func = reinterpret_cast<void*>(conditional_body);
  conditional_kernel.gridDim = dim3(1, 1, 1);
  conditional_kernel.blockDim = dim3(1, 1, 1);
  conditional_kernel.kernelParams = conditional_args;
  hipGraphNode_t body_node = nullptr;
  check(hipGraphAddKernelNode(&body_node, conditional_body_graph, nullptr, 0,
                              &conditional_kernel),
        "hipGraphAddKernelNode(conditional body)");
  hipGraph_t bodies[] = {conditional_body_graph};
  hipGraphNodeParams conditional_params{};
  conditional_params.type = hipGraphNodeTypeConditional;
  conditional_params.conditional.handle = handle;
  conditional_params.conditional.type = hipGraphCondTypeWhile;
  conditional_params.conditional.size = 1;
  conditional_params.conditional.phGraph_out = bodies;
  hipGraphNode_t conditional_node = nullptr;
  check(hipGraphAddNode(&conditional_node, conditional_outer, nullptr, 0,
                        &conditional_params),
        "hipGraphAddNode(conditional)");
  hipGraphExec_t conditional_executable = nullptr;
  check(hipGraphInstantiate(&conditional_executable, conditional_outer, nullptr, nullptr, 0),
        "hipGraphInstantiate(conditional)");

  hipGraph_t static_graph = nullptr;
  check(hipGraphCreate(&static_graph, 0), "hipGraphCreate(static)");
  void* static_args[] = {&counter};
  hipKernelNodeParams static_kernel{};
  static_kernel.func = reinterpret_cast<void*>(static_body);
  static_kernel.gridDim = dim3(1, 1, 1);
  static_kernel.blockDim = dim3(1, 1, 1);
  static_kernel.kernelParams = static_args;
  hipGraphNode_t previous = nullptr;
  for (unsigned int i = 0; i < kIterations; ++i) {
    hipGraphNode_t node = nullptr;
    check(hipGraphAddKernelNode(&node, static_graph,
                                previous == nullptr ? nullptr : &previous,
                                previous == nullptr ? 0 : 1, &static_kernel),
          "hipGraphAddKernelNode(static)");
    previous = node;
  }
  hipGraphExec_t static_executable = nullptr;
  check(hipGraphInstantiate(&static_executable, static_graph, nullptr, nullptr, 0),
        "hipGraphInstantiate(static)");

  const double conditional_ms = average(
      [&] { check(hipGraphLaunch(conditional_executable, nullptr), "hipGraphLaunch(conditional)"); },
      counter, "conditional");
  const double static_ms = average(
      [&] { check(hipGraphLaunch(static_executable, nullptr), "hipGraphLaunch(static)"); },
      counter, "static");
  std::printf("conditional_while_ms=%.6f\nstatic_fixed_graph_ms=%.6f\nratio=%.6f\n",
              conditional_ms, static_ms, conditional_ms / static_ms);

  check(hipGraphExecDestroy(conditional_executable), "hipGraphExecDestroy(conditional)");
  check(hipGraphExecDestroy(static_executable), "hipGraphExecDestroy(static)");
  check(hipGraphDestroy(conditional_outer), "hipGraphDestroy(conditional outer)");
  check(hipGraphDestroy(conditional_body_graph), "hipGraphDestroy(conditional body)");
  check(hipGraphDestroy(static_graph), "hipGraphDestroy(static)");
  check(hipGraphConditionalHandleDestroy(handle), "hipGraphConditionalHandleDestroy");
  check(hipFree(counter), "hipFree(counter)");
  return 0;
}
