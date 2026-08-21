#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxIterations = 100;
constexpr int kTiers[] = {1, 4, 10, 40, 100};

void check(hipError_t status, const char* expression) {
  if (status != hipSuccess) {
    std::cerr << expression << ": " << hipGetErrorString(status) << "\n";
    std::exit(2);
  }
}

#define HIP_CHECK(expr) check((expr), #expr)

__global__ void solver_iteration_kernel(float* state, const int* required, int worlds, int budget) {
  const int world = blockIdx.x * blockDim.x + threadIdx.x;
  if (world >= worlds) return;
  float value = state[world];
  const int needed = required[world];
  for (int iteration = 0; iteration < budget; ++iteration) {
    if (iteration < needed) {
      value = value * 0.997f + 0.003f;
    }
  }
  state[world] = value;
}

struct Graph {
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
};

struct StepTiming {
  std::vector<hipEvent_t> begin;
  std::vector<hipEvent_t> end;
};

StepTiming create_step_timing(int steps) {
  StepTiming timing;
  timing.begin.resize(steps);
  timing.end.resize(steps);
  for (int step = 0; step < steps; ++step) {
    HIP_CHECK(hipEventCreate(&timing.begin[step]));
    HIP_CHECK(hipEventCreate(&timing.end[step]));
  }
  return timing;
}

void destroy_step_timing(StepTiming* timing) {
  for (hipEvent_t event : timing->begin) HIP_CHECK(hipEventDestroy(event));
  for (hipEvent_t event : timing->end) HIP_CHECK(hipEventDestroy(event));
  timing->begin.clear();
  timing->end.clear();
}

std::vector<double> read_step_timing(const StepTiming& timing) {
  std::vector<double> samples;
  samples.reserve(timing.begin.size());
  for (size_t step = 0; step < timing.begin.size(); ++step) {
    float milliseconds = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, timing.begin[step], timing.end[step]));
    samples.push_back(milliseconds);
  }
  return samples;
}

double percentile(std::vector<double> samples, double quantile) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const double index = quantile * static_cast<double>(samples.size() - 1);
  const size_t lower = static_cast<size_t>(index);
  const size_t upper = std::min(lower + 1, samples.size() - 1);
  const double weight = index - static_cast<double>(lower);
  return samples[lower] * (1.0 - weight) + samples[upper] * weight;
}

Graph capture_graph(hipStream_t stream, float* state, const int* required, int worlds, int budget) {
  Graph result;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  hipLaunchKernelGGL(solver_iteration_kernel, dim3((worlds + 255) / 256), dim3(256), 0, stream,
                     state, required, worlds, budget);
  HIP_CHECK(hipStreamEndCapture(stream, &result.graph));
  HIP_CHECK(hipGraphInstantiate(&result.executable, result.graph, nullptr, nullptr, 0));
  return result;
}

void destroy_graph(Graph* graph) {
  if (graph->executable) HIP_CHECK(hipGraphExecDestroy(graph->executable));
  if (graph->graph) HIP_CHECK(hipGraphDestroy(graph->graph));
  graph->executable = nullptr;
  graph->graph = nullptr;
}

int tier_for(int required) {
  for (int tier : kTiers) {
    if (tier >= required) return tier;
  }
  return kMaxIterations;
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point begin,
                  std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_json(const std::string& output, int worlds, int steps, double eager_ms, double static_ms,
                double adaptive_ms, double capture_ms, float max_error, long long eager_work,
                long long adaptive_work, const std::map<int, long long>& histogram,
                const std::vector<double>& eager_samples, const std::vector<double>& static_samples,
                const std::vector<double>& adaptive_samples, int adaptive_graph_count,
                bool conditional_api_detected) {
  std::ofstream file(output);
  file << std::setprecision(10);
  file << "{\n";
  file << "  \"schema\": \"mujoco-warp-amd-hip-graph-v1\",\n";
  file << "  \"worlds\": " << worlds << ",\n";
  file << "  \"steps\": " << steps << ",\n";
  file << "  \"execution\": \"hip_runtime\",\n";
  file << "  \"hip_conditional_graph_api_detected\": "
       << (conditional_api_detected ? "true" : "false") << ",\n";
  file << "  \"capture_ms\": " << capture_ms << ",\n";
  file << "  \"eager_fixed_ms\": " << eager_ms << ",\n";
  file << "  \"static_graph_ms\": " << static_ms << ",\n";
  file << "  \"host_adaptive_ms\": " << adaptive_ms << ",\n";
  file << "  \"host_adaptive_mode\": \"host_selected_tiered_graphs_oracle_schedule\",\n";
  file << "  \"host_adaptive_selection_source\": \"precomputed_required_iterations\",\n";
  file << "  \"host_adaptive_graphs_per_step\": " << adaptive_graph_count << ",\n";
  file << "  \"host_adaptive_runtime_convergence_checks\": 0,\n";
  file << "  \"eager_world_iterations\": " << eager_work << ",\n";
  file << "  \"host_adaptive_world_iterations\": " << adaptive_work << ",\n";
  file << "  \"ideal_iteration_work_reduction\": "
       << (1.0 - static_cast<double>(adaptive_work) / static_cast<double>(eager_work)) << ",\n";
  file << "  \"max_abs_state_error\": " << max_error << ",\n";
  file << "  \"numerically_equivalent\": " << (max_error < 1e-6f ? "true" : "false") << ",\n";
  file << "  \"eager_fixed_gpu_p50_step_ms\": " << percentile(eager_samples, 0.50) << ",\n";
  file << "  \"eager_fixed_gpu_p95_step_ms\": " << percentile(eager_samples, 0.95) << ",\n";
  file << "  \"eager_fixed_gpu_p99_step_ms\": " << percentile(eager_samples, 0.99) << ",\n";
  file << "  \"static_graph_gpu_p50_step_ms\": " << percentile(static_samples, 0.50) << ",\n";
  file << "  \"static_graph_gpu_p95_step_ms\": " << percentile(static_samples, 0.95) << ",\n";
  file << "  \"static_graph_gpu_p99_step_ms\": " << percentile(static_samples, 0.99) << ",\n";
  file << "  \"host_adaptive_gpu_p50_step_ms\": " << percentile(adaptive_samples, 0.50) << ",\n";
  file << "  \"host_adaptive_gpu_p95_step_ms\": " << percentile(adaptive_samples, 0.95) << ",\n";
  file << "  \"host_adaptive_gpu_p99_step_ms\": " << percentile(adaptive_samples, 0.99) << ",\n";
  file << "  \"eager_fixed_worlds_per_second\": "
       << (static_cast<double>(worlds) * steps * 1000.0 / eager_ms) << ",\n";
  file << "  \"static_graph_worlds_per_second\": "
       << (static_cast<double>(worlds) * steps * 1000.0 / static_ms) << ",\n";
  file << "  \"host_adaptive_worlds_per_second\": "
       << (static_cast<double>(worlds) * steps * 1000.0 / adaptive_ms) << ",\n";
  file << "  \"selected_tier_histogram\": {";
  bool first = true;
  for (const auto& [tier, count] : histogram) {
    if (!first) file << ", ";
    first = false;
    file << "\"" << tier << "\": " << count;
  }
  file << "}\n";
  file << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  int worlds = 1024;
  int steps = 1000;
  std::string output = "results/hip_graph_benchmark.json";
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--worlds" && i + 1 < argc) worlds = std::atoi(argv[++i]);
    else if (argument == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    else if (argument == "--output" && i + 1 < argc) output = argv[++i];
  }
  if (worlds <= 0 || steps <= 0) {
    std::cerr << "worlds and steps must be positive\n";
    return 2;
  }

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No HIP device detected\n";
    return 3;
  }
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  std::cout << "device=" << properties.name << "\n";

  std::vector<int> required(worlds);
  for (int world = 0; world < worlds; ++world) {
    required[world] = 1 + ((world * 1103515245 + 12345) & 0x7fffffff) % kMaxIterations;
  }

  int* device_required = nullptr;
  float* eager_state = nullptr;
  float* static_state = nullptr;
  float* adaptive_state = nullptr;
  HIP_CHECK(hipMalloc(&device_required, worlds * sizeof(int)));
  HIP_CHECK(hipMalloc(&eager_state, worlds * sizeof(float)));
  HIP_CHECK(hipMalloc(&static_state, worlds * sizeof(float)));
  HIP_CHECK(hipMalloc(&adaptive_state, worlds * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_required, required.data(), worlds * sizeof(int), hipMemcpyHostToDevice));

  std::vector<float> initial(worlds, 1.0f);
  HIP_CHECK(hipMemcpy(eager_state, initial.data(), worlds * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(static_state, initial.data(), worlds * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(adaptive_state, initial.data(), worlds * sizeof(float), hipMemcpyHostToDevice));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  const auto eager_begin = std::chrono::high_resolution_clock::now();
  StepTiming eager_timing = create_step_timing(steps);
  for (int step = 0; step < steps; ++step) {
    HIP_CHECK(hipEventRecord(eager_timing.begin[step], stream));
    hipLaunchKernelGGL(solver_iteration_kernel, dim3((worlds + 255) / 256), dim3(256), 0, stream,
                       eager_state, device_required, worlds, kMaxIterations);
    HIP_CHECK(hipEventRecord(eager_timing.end[step], stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto eager_end = std::chrono::high_resolution_clock::now();
  std::vector<double> eager_samples = read_step_timing(eager_timing);

  const auto capture_begin = std::chrono::high_resolution_clock::now();
  Graph static_graph = capture_graph(stream, static_state, device_required, worlds, kMaxIterations);
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto capture_end = std::chrono::high_resolution_clock::now();

  const auto static_begin = std::chrono::high_resolution_clock::now();
  StepTiming static_timing = create_step_timing(steps);
  for (int step = 0; step < steps; ++step) {
    HIP_CHECK(hipEventRecord(static_timing.begin[step], stream));
    HIP_CHECK(hipGraphLaunch(static_graph.executable, stream));
    HIP_CHECK(hipEventRecord(static_timing.end[step], stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto static_end = std::chrono::high_resolution_clock::now();
  std::vector<double> static_samples = read_step_timing(static_timing);

  std::map<int, Graph> graphs;
  std::vector<int> grouped_required;
  std::vector<int> grouped_original_index;
  std::map<int, int> group_offsets;
  std::map<int, int> group_counts;
  grouped_required.reserve(worlds);
  grouped_original_index.reserve(worlds);
  for (int tier : kTiers) {
    group_offsets[tier] = static_cast<int>(grouped_required.size());
    for (int world = 0; world < worlds; ++world) {
      if (tier_for(required[world]) != tier) continue;
      grouped_required.push_back(required[world]);
      grouped_original_index.push_back(world);
    }
    group_counts[tier] = static_cast<int>(grouped_required.size()) - group_offsets[tier];
  }

  int* device_group_required = nullptr;
  float* grouped_state = nullptr;
  HIP_CHECK(hipMalloc(&device_group_required, worlds * sizeof(int)));
  HIP_CHECK(hipMalloc(&grouped_state, worlds * sizeof(float)));
  std::vector<float> grouped_initial(worlds, 1.0f);
  HIP_CHECK(hipMemcpy(device_group_required, grouped_required.data(),
                      worlds * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(grouped_state, grouped_initial.data(),
                      worlds * sizeof(float), hipMemcpyHostToDevice));

  const auto adaptive_capture_begin = std::chrono::high_resolution_clock::now();
  for (int tier : kTiers) {
    if (group_counts[tier] == 0) continue;
    graphs.emplace(tier, capture_graph(stream, grouped_state + group_offsets[tier],
                                       device_group_required + group_offsets[tier],
                                       group_counts[tier], tier));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto adaptive_capture_end = std::chrono::high_resolution_clock::now();

  std::map<int, long long> histogram;
  long long adaptive_work = 0;
  for (int tier : kTiers) {
    if (group_counts[tier] == 0) continue;
    histogram[tier] = static_cast<long long>(group_counts[tier]) * steps;
    adaptive_work += static_cast<long long>(tier) * group_counts[tier] * steps;
  }
  const auto adaptive_begin = std::chrono::high_resolution_clock::now();
  StepTiming adaptive_timing = create_step_timing(steps);
  for (int step = 0; step < steps; ++step) {
    HIP_CHECK(hipEventRecord(adaptive_timing.begin[step], stream));
    for (int tier : kTiers) {
      if (group_counts[tier] == 0) continue;
      HIP_CHECK(hipGraphLaunch(graphs.at(tier).executable, stream));
    }
    HIP_CHECK(hipEventRecord(adaptive_timing.end[step], stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto adaptive_end = std::chrono::high_resolution_clock::now();
  std::vector<double> adaptive_samples = read_step_timing(adaptive_timing);

  std::vector<float> eager_host(worlds), static_host(worlds), grouped_host(worlds), adaptive_host(worlds);
  HIP_CHECK(hipMemcpy(eager_host.data(), eager_state, worlds * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(static_host.data(), static_state, worlds * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(grouped_host.data(), grouped_state, worlds * sizeof(float), hipMemcpyDeviceToHost));
  for (int grouped = 0; grouped < worlds; ++grouped) {
    adaptive_host[grouped_original_index[grouped]] = grouped_host[grouped];
  }
  float max_error = 0.0f;
  for (int world = 0; world < worlds; ++world) {
    max_error = std::max(max_error, std::fabs(eager_host[world] - static_host[world]));
    max_error = std::max(max_error, std::fabs(eager_host[world] - adaptive_host[world]));
  }

  const long long eager_work = static_cast<long long>(kMaxIterations) * worlds * steps;
  const double capture_ms = elapsed_ms(capture_begin, capture_end) +
                            elapsed_ms(adaptive_capture_begin, adaptive_capture_end);
  print_json(output, worlds, steps, elapsed_ms(eager_begin, eager_end), elapsed_ms(static_begin, static_end),
             elapsed_ms(adaptive_begin, adaptive_end), capture_ms, max_error, eager_work, adaptive_work,
             histogram, eager_samples, static_samples, adaptive_samples,
             static_cast<int>(graphs.size()), false);
  std::cout << "wrote=" << output << "\n";

  destroy_graph(&static_graph);
  for (auto& [tier, graph] : graphs) destroy_graph(&graph);
  destroy_step_timing(&eager_timing);
  destroy_step_timing(&static_timing);
  destroy_step_timing(&adaptive_timing);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_required));
  HIP_CHECK(hipFree(eager_state));
  HIP_CHECK(hipFree(static_state));
  HIP_CHECK(hipFree(adaptive_state));
  HIP_CHECK(hipFree(device_group_required));
  HIP_CHECK(hipFree(grouped_state));
  return 0;
}
