#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void advance_condition(unsigned int* counter,
                                  hipGraphConditionalHandle handle,
                                  unsigned int limit) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const unsigned int next = atomicAdd(counter, 1) + 1;
    hipGraphSetConditional(handle, next < limit ? 1u : 0u);
  }
}

static void check(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    std::exit(1);
  }
}

int main() {
  constexpr unsigned int kLimit = 5;
  unsigned int* counter = nullptr;
  check(hipMalloc(&counter, sizeof(*counter)), "hipMalloc(counter)");
  check(hipMemset(counter, 0, sizeof(*counter)), "hipMemset(counter)");

  hipGraph_t outer = nullptr;
  hipGraph_t body = nullptr;
  check(hipGraphCreate(&outer, 0), "hipGraphCreate(outer)");
  check(hipGraphCreate(&body, 0), "hipGraphCreate(body)");

  hipGraphConditionalHandle handle = 0;
  check(hipGraphConditionalHandleCreate(&handle, outer, 1, hipGraphCondAssignDefault),
        "hipGraphConditionalHandleCreate");

  hipGraphNode_t body_node = nullptr;
  unsigned int limit = kLimit;
  void* kernel_args[] = {&counter, &handle, &limit};
  hipKernelNodeParams kernel_params{};
  kernel_params.func = reinterpret_cast<void*>(advance_condition);
  kernel_params.gridDim = dim3(1, 1, 1);
  kernel_params.blockDim = dim3(1, 1, 1);
  kernel_params.kernelParams = kernel_args;
  check(hipGraphAddKernelNode(&body_node, body, nullptr, 0, &kernel_params),
        "hipGraphAddKernelNode(body)");

  hipGraph_t body_graphs[] = {body};
  hipGraphNodeParams conditional_params{};
  conditional_params.type = hipGraphNodeTypeConditional;
  conditional_params.conditional.handle = handle;
  conditional_params.conditional.type = hipGraphCondTypeWhile;
  conditional_params.conditional.size = 1;
  conditional_params.conditional.phGraph_out = body_graphs;

  hipGraphNode_t conditional_node = nullptr;
  check(hipGraphAddNode(&conditional_node, outer, nullptr, 0, &conditional_params),
        "hipGraphAddNode(conditional)");

  hipGraphExec_t executable = nullptr;
  check(hipGraphInstantiate(&executable, outer, nullptr, nullptr, 0),
        "hipGraphInstantiate");
  check(hipGraphLaunch(executable, nullptr), "hipGraphLaunch");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");

  unsigned int result = 0;
  check(hipMemcpy(&result, counter, sizeof(result), hipMemcpyDeviceToHost),
        "hipMemcpy(counter)");
  std::printf("conditional_while_device counter=%u expected=%u\n", result, kLimit);

  check(hipGraphExecDestroy(executable), "hipGraphExecDestroy");
  check(hipGraphDestroy(outer), "hipGraphDestroy(outer)");
  check(hipGraphDestroy(body), "hipGraphDestroy(body)");
  check(hipGraphConditionalHandleDestroy(handle), "hipGraphConditionalHandleDestroy");
  check(hipFree(counter), "hipFree(counter)");
  return result == kLimit ? 0 : 2;
}
