#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <rapidcsv.h>
#include <cxxopts.hpp>

#ifdef REDUCE_USE_CUDA
#include "whip/cuda/whip.hpp"
#include <curand.h>
#else
#include "whip/hip/whip.hpp"
#include <hiprand/hiprand.h>
#endif
#include "mdarray.hpp"
#include "timing/rt_graph.hpp"
#include "utils.hpp"


template <typename T, int repetition = 8> __global__ void KendallTau(const T*__restrict__ X__,
                                                 const T*__restrict__ Y__,
                                                 const int starting_position__,
                                                 const int m,
                                                 const int N,
                                                 int *C,
                                                 int *D,
                                                 int* ties_x,
                                                 int* ties_y)
{
  if ((blockIdx.z  + starting_position__) >= m * m)
    return;

  const int row = (blockIdx.z + starting_position__) % m;
  const int col = (blockIdx.z + starting_position__) / m;

  if (row < col) {
    return;
  }


  int Cb = 0;
  int Db = 0;
  int tx = 0;
  int ty = 0;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

  __shared__ T xj[64], yj[64];

  if (tid + repetition * blockIdx.y * blockDim.y < N) {
    xj[tid] = X__[tid + repetition * blockIdx.y * blockDim.y + row * N];
    yj[tid] = Y__[tid + repetition * blockIdx.y * blockDim.y + col * N];
  } else {
    xj[tid] = 0;
    yj[tid] = 0;
  }

  __syncthreads();

  for (int l = 0; l < repetition; l++) {
    const int j = threadIdx.y + (repetition * blockIdx.y + l) * blockDim.y;
    if (j < N) {
      // const T xj = X__[j + row * N];
      // const T yj = Y__[j + col * N];
      for (int k = 0; k < repetition; k++) {
        const int i = threadIdx.x + (repetition * blockIdx.x + k) * blockDim.x;
        if ((i < N) && (i < j)) {
          const T xi = X__[i + row * N];
          const T yi = Y__[i + col * N];
          int x_comp = (xi > xj[threadIdx.y + l * repetition]) - (xi < xj[threadIdx.y + l * repetition]);
          int y_comp = (yi > yj[threadIdx.y + l * repetition]) - (yi < yj[threadIdx.y + l * repetition]);
          int product = x_comp * y_comp;
          if (product > 0)
            Cb++;
          if (product < 0)
            Db++;
          if (x_comp == 0)
            tx++;
          if (y_comp == 0)
            ty++;
        }
      }
    }
  }

  __shared__ int accumulator[64];
  //  const int index = row * (row + 1) / 2 + col;

  if (C != nullptr) {
    accumulator[tid] = Cb;
    __syncthreads();
    if (tid < 32)
      accumulator[tid] += accumulator[tid + 32];
    __syncthreads();
    if (tid < 16)
      accumulator[tid] += accumulator[tid + 16];
    __syncthreads();
    if (tid < 8)
      accumulator[tid] += accumulator[tid + 8];
    __syncthreads();
    if (tid < 4)
      accumulator[tid] += accumulator[tid + 4];
    __syncthreads();
    if (tid < 2)
      accumulator[tid] += accumulator[tid + 2];
    __syncthreads();
    if (tid < 1)
      accumulator[tid] += accumulator[tid + 1];
    __syncthreads();
    if ((tid == 0) && (accumulator[0] != 0))
      atomicAdd(C + blockIdx.z, accumulator[0]);
  }

  if (D != nullptr) {
    accumulator[tid] = Db;
    __syncthreads();
    if (tid < 32)
      accumulator[tid] += accumulator[tid + 32];
    __syncthreads();
    if (tid < 16)
      accumulator[tid] += accumulator[tid + 16];
    __syncthreads();
    if (tid < 8)
      accumulator[tid] += accumulator[tid + 8];
    __syncthreads();
    if (tid < 4)
      accumulator[tid] += accumulator[tid + 4];
    __syncthreads();
    if (tid < 2)
      accumulator[tid] += accumulator[tid + 2];
    __syncthreads();
    if (tid < 1)
      accumulator[tid] += accumulator[tid + 1];
    __syncthreads();

    if ((tid == 0) && (accumulator[0] != 0))
    atomicAdd(D + blockIdx.z, accumulator[0]);
  }

  if (ties_x != nullptr) {
    accumulator[tid] = tx;
    __syncthreads();
    if (tid < 32)
      accumulator[tid] += accumulator[tid + 32];
    __syncthreads();
    if (tid < 16)
      accumulator[tid] += accumulator[tid + 16];
    __syncthreads();
    if (tid < 8)
      accumulator[tid] += accumulator[tid + 8];
    __syncthreads();
    if (tid < 4)
      accumulator[tid] += accumulator[tid + 4];
    __syncthreads();
    if (tid < 2)
      accumulator[tid] += accumulator[tid + 2];
    __syncthreads();
    if (tid < 1)
      accumulator[tid] += accumulator[tid + 1];
    __syncthreads();
    if ((tid == 0) && (accumulator[0] != 0))
      atomicAdd(ties_x  + blockIdx.z, accumulator[0]);
  }

  if (ties_y != nullptr) {
    accumulator[tid] = ty;
    __syncthreads();
    if (tid < 32)
      accumulator[tid] += accumulator[tid + 32];
    __syncthreads();
    if (tid < 16)
      accumulator[tid] += accumulator[tid + 16];
    __syncthreads();
    if (tid < 8)
      accumulator[tid] += accumulator[tid + 8];
    __syncthreads();
    if (tid < 4)
      accumulator[tid] += accumulator[tid + 4];
    __syncthreads();
    if (tid < 2)
      accumulator[tid] += accumulator[tid + 2];
    __syncthreads();
    if (tid < 1)
      accumulator[tid] += accumulator[tid + 1];
    __syncthreads();
    if ((tid == 0) && (accumulator[0] != 0))
      atomicAdd(ties_y + blockIdx.z, accumulator[0]);
  }
}


template <typename T, int repetition = 8, int block_size = 256> __host__ void KendalTauGPU(const T*__restrict__ X__, const T* __restrict__ Y__, const int m, const int N, double * results__)
{
  const int nstep = (m * m + block_size - 1) / block_size;
  const int Nblock = (N + 8 * repetition - 1) / (8 * repetition);

  const dim3 block_dim = dim3(8, 8, 1);
  const dim3 grid_dim = dim3(Nblock, Nblock, block_size);

  const double num_pair = N * (N - 1) / 2;
  std::vector<double> res(block_size);

  mdarray<int, 2, CblasRowMajor> res_(4, block_size);
  res_.allocate(memory_t::device);
  for (int i = 0; i < nstep; i++) {
    whip::memset(res_.at<device_t::GPU>(), 0, sizeof(int) * 4 * block_size);
    KendallTau<T, repetition><<<grid_dim, block_dim>>>(X__,
                                                       Y__,
                                                       i * block_size,
                                                       m,
                                                       N,
                                                       res_.at<device_t::GPU>(),
                                                       res_.at<device_t::GPU>() + block_size,
                                                       res_.at<device_t::GPU>() + 2 * block_size,
                                                       res_.at<device_t::GPU>() + 3 * block_size);
    
    res_.copy<memory_t::device, memory_t::host>();

    for (int j = 0; j < block_size; j++) {
      res[j] = static_cast<double>(res_[j] - res_[j + block_size]) / num_pair;
      const int row = (j + i * block_size) % m;
      const int col = (j + i * block_size) / m;
      if ((row < m) && (col < m) && (row >= col)) {
        results__[row * m + col] = res[j];
      }
    }
  }
  res_.clear();
}

int main(int argc, char **argv)
{
  rt_graph::Timer timer_;
  cxxopts::Options options("KendallTau", "Compute the KendallTau correlations of a given csv file");
  options.add_options()
    ("d,debug", "Enable debugging") // a bool parameter
    ("n,length", "Length of a given list", cxxopts::value<int>()->default_value("1"))
    ("m,nlists", "number of lists", cxxopts::value<int>()->default_value("1"))
    ("file", "list of two csv files to be compared to. If only one file is provided then kendallTau will calculate correlation within the same table", cxxopts::value<std::vector<std::string>>())
    ("o,output", "name of the csv output file", cxxopts::value<std::string>())
    ("h,help", "Print usage")
    ;


  options.parse_positional({"file"});
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  auto &file_name = result["file"].as<std::vector<std::string>>();
  std::string output = result["output"].as<std::string>();
  const int nlists = result["nlists"].as<int>();
  const int length = result["length"].as<int>();

  mdarray<int, 2, CblasRowMajor> data1_(nlists, length);
  mdarray<int, 2, CblasRowMajor> data2_(nlists, length);
  mdarray<double, 2, CblasRowMajor> results(nlists, nlists);
  rapidcsv::Document doc(file_name[0], rapidcsv::LabelParams(-1, -1));

  for (int i = 0; i < nlists; i++) {
    std::vector<int> close = doc.GetRow<int>(i);
    memcpy(data1_.at<device_t::CPU>(i, 0), close.data(), sizeof(int) * close.size());
  }

  if(file_name.size() == 2) {
    rapidcsv::Document doc1(file_name[1], rapidcsv::LabelParams(-1, -1));
    for (int i = 0; i < nlists; i++) {
      std::vector<int> close = doc.GetRow<int>(i);
      memcpy(data2_.at<device_t::CPU>(i, 0), close.data(), sizeof(int) * close.size());
    }
  } else {
    memcpy(data2_.at<device_t::CPU>(), data1_.at<device_t::CPU>(), data1_.size() * sizeof(int));
  }

  data1_.allocate(memory_t::device);
  data2_.allocate(memory_t::device);
  results.allocate(memory_t::device);
  data1_.copy<memory_t::host, memory_t::device>();
  data2_.copy<memory_t::host, memory_t::device>();
  timer_.start("calculate_kendall_tau");
  KendalTauGPU<int>(data1_.at<device_t::GPU>(),
                    data2_.at<device_t::GPU>(),
                    nlists,
                    length,
                    results.at<device_t::CPU>());
  timer_.stop("calculate_kendall_tau");

  for (int  i = 0; i < 10; i++) {
    for (int  j = 0; j < 10; j++) {
      printf("%.5lf ", results(i,j));
    }
    printf("\n");
  }

  FILE *f = fopen(output.c_str(), "w+");
  for (int  i = 1; i < nlists; i++) {
    for (int  j = 0; j < i - 1; j++) {
      fprintf(f, "%.15lf,", results(i, j));
    }
    if (i == nlists - 1)
      fprintf(f, "%.15lf", results(i, i - 1));
    else
      fprintf(f, "%.15lf,", results(i, i - 1));
  }
  fclose(f);

  auto timing_result = timer_.process();
  std::cout << timing_result.print(
                                   {rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                    rt_graph::Stat::Median, rt_graph::Stat::Min, rt_graph::Stat::Max});
  results.clear();
  data1_.clear();
  data2_.clear();
  return 0;
}
