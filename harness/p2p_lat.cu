// Per-pair GPU D2D latency + bandwidth, peer DMA vs host-staged.
// Output CSV: mode,src,dst,lat_us,bw_gbs
// Build: nvcc -arch=sm_61 -O2 -o p2p_lat p2p_lat.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static const size_t BW_BYTES  = 128ull << 20;  // 128 MiB
static const int    BW_REPS   = 20;
static const int    LAT_REPS  = 2000;

int main() {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    void *buf[16];
    for (int i = 0; i < n; i++) {
        CK(cudaSetDevice(i));
        CK(cudaMalloc(&buf[i], BW_BYTES));
    }

    // mode 0 = host-staged (peer access never enabled)
    // mode 1 = peer DMA    (enable for all pairs, then re-measure)
    for (int mode = 0; mode <= 1; mode++) {
        if (mode == 1) {
            for (int i = 0; i < n; i++) {
                CK(cudaSetDevice(i));
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    int can = 0;
                    CK(cudaDeviceCanAccessPeer(&can, i, j));
                    if (can) CK(cudaDeviceEnablePeerAccess(j, 0));
                }
            }
        }
        for (int s = 0; s < n; s++) {
            CK(cudaSetDevice(s));
            cudaStream_t st;
            CK(cudaStreamCreate(&st));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            for (int d = 0; d < n; d++) {
                if (s == d) continue;
                // warmup
                CK(cudaMemcpyPeerAsync(buf[d], d, buf[s], s, 4, st));
                CK(cudaStreamSynchronize(st));
                // latency: tiny copies, sync each (includes launch overhead; comparative)
                CK(cudaEventRecord(e0, st));
                for (int r = 0; r < LAT_REPS; r++)
                    CK(cudaMemcpyPeerAsync(buf[d], d, buf[s], s, 4, st));
                CK(cudaEventRecord(e1, st));
                CK(cudaStreamSynchronize(st));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                double lat_us = ms * 1000.0 / LAT_REPS;
                // bandwidth: large copies
                CK(cudaMemcpyPeerAsync(buf[d], d, buf[s], s, BW_BYTES, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaEventRecord(e0, st));
                for (int r = 0; r < BW_REPS; r++)
                    CK(cudaMemcpyPeerAsync(buf[d], d, buf[s], s, BW_BYTES, st));
                CK(cudaEventRecord(e1, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaEventElapsedTime(&ms, e0, e1));
                double bw = (double)BW_BYTES * BW_REPS / (ms / 1000.0) / 1e9;
                printf("%s,%d,%d,%.2f,%.2f\n",
                       mode ? "peer" : "staged", s, d, lat_us, bw);
                fflush(stdout);
            }
            CK(cudaEventDestroy(e0));
            CK(cudaEventDestroy(e1));
            CK(cudaStreamDestroy(st));
        }
    }
    return 0;
}
