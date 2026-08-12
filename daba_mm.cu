// CUDA implementation of decoupled majorization-minimization (MM) bundle adjustment,
// DABA-style (Fan et al., RSS 2023) single-GPU / shared-memory core.
//
// Ported from reference_mm.py (this directory), which is validated against all four
// Section-6 acceptance criteria on ladybug-49. Every formula here (projection,
// analytic Jacobians, MM block solve, Nesterov extrapolation with cost-based restart)
// mirrors that reference exactly -- see reference_mm.py for the derivation/validation
// notes (in particular the Jacobian gradient-check eps tuning, and the q-indexing
// convention for the beta=(q-1)/(q+2) momentum schedule).
//
// Precision: fp64 (Scalar=double) throughout, per the spec's stated priority
// ("fp32 stalls above the true minimum" -- correctness first). fp32 is not
// implemented here; Scalar is written as a single typedef specifically so switching
// it is a one-line change, but that path is untested.
//
// Determinism: this version accumulates per-camera/per-point normal-equation blocks
// via atomicAdd (correctness-first, per the spec's explicit allowance -- "atomics are
// fine as a first correct version"). Run-to-run bitwise reproducibility is not
// guaranteed; the acceptance criteria use tolerant thresholds (4 sig figs, +/-0.5%)
// that atomic-order fp64 noise does not threaten. A sorted/segmented-reduction
// deterministic mode is future work (see README).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using Scalar = double;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// ============================================================== BAL loading
struct BalData {
  int ncam = 0, npt = 0, nobs = 0;
  std::vector<int> cam_idx, pt_idx;
  std::vector<Scalar> uv;    // 2*nobs
  std::vector<Scalar> cams;  // 9*ncam: angle-axis(3), t(3), f, k1, k2
  std::vector<Scalar> pts;   // 3*npt
};

// Streams the numeric tokens directly into typed arrays (no intermediate string-list
// construction, no per-observation objects) so this scales to venice-1778 (~280MB).
BalData LoadBal(const std::string& path) {
  std::ifstream fh(path);
  if (!fh) {
    std::fprintf(stderr, "Failed to open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  BalData d;
  fh >> d.ncam >> d.npt >> d.nobs;
  d.cam_idx.resize(d.nobs);
  d.pt_idx.resize(d.nobs);
  d.uv.resize(2 * d.nobs);
  for (int i = 0; i < d.nobs; ++i) {
    fh >> d.cam_idx[i] >> d.pt_idx[i] >> d.uv[2 * i] >> d.uv[2 * i + 1];
  }
  d.cams.resize(9 * d.ncam);
  for (auto& v : d.cams) fh >> v;
  d.pts.resize(3 * d.npt);
  for (auto& v : d.pts) fh >> v;
  return d;
}

// ============================================================== device math helpers
// Angle-axis -> 3x3 rotation matrix (Rodrigues), row-major output.
__device__ __forceinline__ void ExpSO3(const Scalar* a, Scalar* R) {
  Scalar th = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  if (th < 1e-12) {
    R[0] = 1; R[1] = 0; R[2] = 0;
    R[3] = 0; R[4] = 1; R[5] = 0;
    R[6] = 0; R[7] = 0; R[8] = 1;
    return;
  }
  Scalar kx = a[0] / th, ky = a[1] / th, kz = a[2] / th;
  Scalar c = std::cos(th), s = std::sin(th), C = 1 - c;
  R[0] = c + kx * kx * C;      R[1] = kx * ky * C - kz * s; R[2] = kx * kz * C + ky * s;
  R[3] = ky * kx * C + kz * s; R[4] = c + ky * ky * C;      R[5] = ky * kz * C - kx * s;
  R[6] = kz * kx * C - ky * s; R[7] = kz * ky * C + kx * s; R[8] = c + kz * kz * C;
}

// 3x3 rotation matrix -> angle-axis (SO(3) log), robust near theta=0 and theta=pi.
__device__ __forceinline__ void LogSO3(const Scalar* R, Scalar* a) {
  Scalar tr = R[0] + R[4] + R[8];
  Scalar cos_th = 0.5 * (tr - 1.0);
  cos_th = cos_th > 1.0 ? 1.0 : (cos_th < -1.0 ? -1.0 : cos_th);
  Scalar th = std::acos(cos_th);
  Scalar vx = R[7] - R[5], vy = R[2] - R[6], vz = R[3] - R[1];
  if (th < 1e-8) {
    a[0] = 0.5 * vx; a[1] = 0.5 * vy; a[2] = 0.5 * vz;
    return;
  }
  if (th > M_PI - 1e-6) {
    // near pi: vee(R-R^T)~0, fall back to the R+I diagonal (not expected to trigger
    // in this MM loop, since poses move a small angle per outer iteration).
    Scalar dx = R[0] + 1.0, dy = R[4] + 1.0, dz = R[8] + 1.0;
    dx = dx > 0 ? dx : 0; dy = dy > 0 ? dy : 0; dz = dz > 0 ? dz : 0;
    Scalar nx = std::sqrt(dx), ny = std::sqrt(dy), nz = std::sqrt(dz);
    Scalar n = std::sqrt(nx * nx + ny * ny + nz * nz) + 1e-12;
    a[0] = nx / n * th; a[1] = ny / n * th; a[2] = nz / n * th;
    return;
  }
  Scalar coef = th / (2.0 * std::sin(th));
  a[0] = coef * vx; a[1] = coef * vy; a[2] = coef * vz;
}

// out(3x3) = A(3x3) @ B(3x3), row-major.
__device__ __forceinline__ void Mat3Mul(const Scalar* A, const Scalar* B, Scalar* out) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Scalar s = 0;
      for (int k = 0; k < 3; ++k) s += A[3 * i + k] * B[3 * k + j];
      out[3 * i + j] = s;
    }
  }
}
// out(3x3) = A(3x3) @ B(3x3)^T.
__device__ __forceinline__ void Mat3MulT(const Scalar* A, const Scalar* B, Scalar* out) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Scalar s = 0;
      for (int k = 0; k < 3; ++k) s += A[3 * i + k] * B[3 * j + k];
      out[3 * i + j] = s;
    }
  }
}

// In-place Cholesky solve of a small SPD system H x = b (H is N x N, row-major,
// overwritten with its Cholesky factor L). Single thread, N in {3,6}; N is small
// enough that a plain triple loop is fine (no shared memory / warp cooperation).
template <int N>
__device__ __forceinline__ void CholeskySolve(Scalar H[N][N], Scalar b[N], Scalar x[N]) {
  Scalar L[N][N];
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) L[i][j] = 0;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    Scalar s = H[j][j];
#pragma unroll
    for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
    L[j][j] = std::sqrt(s > 1e-300 ? s : 1e-300);
#pragma unroll
    for (int i = j + 1; i < N; ++i) {
      Scalar s2 = H[i][j];
#pragma unroll
      for (int k = 0; k < j; ++k) s2 -= L[i][k] * L[j][k];
      L[i][j] = s2 / L[j][j];
    }
  }
  Scalar y[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    Scalar s = b[i];
#pragma unroll
    for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
#pragma unroll
  for (int i = N - 1; i >= 0; --i) {
    Scalar s = y[i];
#pragma unroll
    for (int k = i + 1; k < N; ++k) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
}

// ============================================================== Kernel A: Jacobian + accumulate
// One thread per observation. Computes the residual and the analytic Jacobian blocks
// (Section 2 of the spec), then atomically accumulates J^T J (6x6 for the camera block,
// 3x3 for the point block) and J^T r into per-camera/per-point global buffers, plus the
// per-observation squared residual into a single running cost total.
__global__ void KernelJacobianAccumulate(
    const int* __restrict__ cam_idx, const int* __restrict__ pt_idx,
    const Scalar* __restrict__ uv, const Scalar* __restrict__ R,
    const Scalar* __restrict__ t, const Scalar* __restrict__ X,
    const Scalar* __restrict__ f, const Scalar* __restrict__ k1,
    const Scalar* __restrict__ k2, int nobs,
    Scalar* __restrict__ Hc, Scalar* __restrict__ gc,  // NC*36, NC*6
    Scalar* __restrict__ Hp, Scalar* __restrict__ gp,  // NP*9,  NP*3
    Scalar* __restrict__ cost_out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= nobs) return;
  int c = cam_idx[o], p = pt_idx[o];
  const Scalar* Rc = R + 9 * c;
  Scalar Xp0 = X[3 * p], Xp1 = X[3 * p + 1], Xp2 = X[3 * p + 2];
  Scalar rx = Rc[0] * Xp0 + Rc[1] * Xp1 + Rc[2] * Xp2;
  Scalar ry = Rc[3] * Xp0 + Rc[4] * Xp1 + Rc[5] * Xp2;
  Scalar rz = Rc[6] * Xp0 + Rc[7] * Xp1 + Rc[8] * Xp2;
  Scalar Px = rx + t[3 * c], Py = ry + t[3 * c + 1], Pz = rz + t[3 * c + 2];
  Scalar xp = -Px / Pz, yp = -Py / Pz;
  Scalar r2 = xp * xp + yp * yp;
  Scalar fo = f[c], k1o = k1[c], k2o = k2[c];
  Scalar dist = 1.0 + k1o * r2 + k2o * r2 * r2;
  Scalar res0 = fo * dist * xp - uv[2 * o];
  Scalar res1 = fo * dist * yp - uv[2 * o + 1];

  atomicAdd(cost_out, 0.5 * (res0 * res0 + res1 * res1));

  Scalar dp00 = -1.0 / Pz, dp02 = Px / (Pz * Pz);
  Scalar dp11 = -1.0 / Pz, dp12 = Py / (Pz * Pz);
  Scalar g = k1o + 2.0 * k2o * r2;
  Scalar Jd00 = fo * (dist + 2.0 * xp * xp * g), Jd01 = fo * (2.0 * xp * yp * g);
  Scalar Jd10 = Jd01, Jd11 = fo * (dist + 2.0 * yp * yp * g);

  // Jpc (2x3) = Jd @ dp
  Scalar Jpc00 = Jd00 * dp00, Jpc01 = Jd01 * dp11, Jpc02 = Jd00 * dp02 + Jd01 * dp12;
  Scalar Jpc10 = Jd10 * dp00, Jpc11 = Jd11 * dp11, Jpc12 = Jd10 * dp02 + Jd11 * dp12;

  // J_pt (2x3) = Jpc @ Rc
  Scalar Jpt00 = Jpc00 * Rc[0] + Jpc01 * Rc[3] + Jpc02 * Rc[6];
  Scalar Jpt01 = Jpc00 * Rc[1] + Jpc01 * Rc[4] + Jpc02 * Rc[7];
  Scalar Jpt02 = Jpc00 * Rc[2] + Jpc01 * Rc[5] + Jpc02 * Rc[8];
  Scalar Jpt10 = Jpc10 * Rc[0] + Jpc11 * Rc[3] + Jpc12 * Rc[6];
  Scalar Jpt11 = Jpc10 * Rc[1] + Jpc11 * Rc[4] + Jpc12 * Rc[7];
  Scalar Jpt12 = Jpc10 * Rc[2] + Jpc11 * Rc[5] + Jpc12 * Rc[8];

  // J_rot (2x3) = Jpc @ (-skew(Prot)), skew(Prot)=[[0,-rz,ry],[rz,0,-rx],[-ry,rx,0]]
  Scalar Jrot00 = -Jpc01 * rz + Jpc02 * ry;
  Scalar Jrot01 = Jpc00 * rz - Jpc02 * rx;
  Scalar Jrot02 = -Jpc00 * ry + Jpc01 * rx;
  Scalar Jrot10 = -Jpc11 * rz + Jpc12 * ry;
  Scalar Jrot11 = Jpc10 * rz - Jpc12 * rx;
  Scalar Jrot12 = -Jpc10 * ry + Jpc11 * rx;

  // J_cam (2x6) = [J_rot | Jpc]  (translation block IS Jpc, per spec)
  Scalar Jcam[2][6] = {{Jrot00, Jrot01, Jrot02, Jpc00, Jpc01, Jpc02},
                        {Jrot10, Jrot11, Jrot12, Jpc10, Jpc11, Jpc12}};
  Scalar Jpt[2][3] = {{Jpt00, Jpt01, Jpt02}, {Jpt10, Jpt11, Jpt12}};
  Scalar res[2] = {res0, res1};

  Scalar* HcC = Hc + 36 * c;
  Scalar* gcC = gc + 6 * c;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    Scalar gi = Jcam[0][i] * res[0] + Jcam[1][i] * res[1];
    atomicAdd(&gcC[i], gi);
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      Scalar hij = Jcam[0][i] * Jcam[0][j] + Jcam[1][i] * Jcam[1][j];
      atomicAdd(&HcC[6 * i + j], hij);
    }
  }
  Scalar* HpP = Hp + 9 * p;
  Scalar* gpP = gp + 3 * p;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    Scalar gi = Jpt[0][i] * res[0] + Jpt[1][i] * res[1];
    atomicAdd(&gpP[i], gi);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      Scalar hij = Jpt[0][i] * Jpt[0][j] + Jpt[1][i] * Jpt[1][j];
      atomicAdd(&HpP[3 * i + j], hij);
    }
  }
}

// ============================================================== Kernel: cost-only
__global__ void KernelCostOnly(const int* __restrict__ cam_idx,
                                const int* __restrict__ pt_idx,
                                const Scalar* __restrict__ uv,
                                const Scalar* __restrict__ R,
                                const Scalar* __restrict__ t,
                                const Scalar* __restrict__ X,
                                const Scalar* __restrict__ f,
                                const Scalar* __restrict__ k1,
                                const Scalar* __restrict__ k2, int nobs,
                                Scalar* __restrict__ cost_out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= nobs) return;
  int c = cam_idx[o], p = pt_idx[o];
  const Scalar* Rc = R + 9 * c;
  Scalar Xp0 = X[3 * p], Xp1 = X[3 * p + 1], Xp2 = X[3 * p + 2];
  Scalar rx = Rc[0] * Xp0 + Rc[1] * Xp1 + Rc[2] * Xp2;
  Scalar ry = Rc[3] * Xp0 + Rc[4] * Xp1 + Rc[5] * Xp2;
  Scalar rz = Rc[6] * Xp0 + Rc[7] * Xp1 + Rc[8] * Xp2;
  Scalar Px = rx + t[3 * c], Py = ry + t[3 * c + 1], Pz = rz + t[3 * c + 2];
  Scalar xp = -Px / Pz, yp = -Py / Pz;
  Scalar r2 = xp * xp + yp * yp;
  Scalar fo = f[c], k1o = k1[c], k2o = k2[c];
  Scalar dist = 1.0 + k1o * r2 + k2o * r2 * r2;
  Scalar res0 = fo * dist * xp - uv[2 * o];
  Scalar res1 = fo * dist * yp - uv[2 * o + 1];
  atomicAdd(cost_out, 0.5 * (res0 * res0 + res1 * res1));
}

// ============================================================== Kernel B+C: block solve + retract
__global__ void KernelSolveRetractCameras(const Scalar* __restrict__ Hc_raw,
                                           const Scalar* __restrict__ gc,
                                           const Scalar* __restrict__ R_in,
                                           const Scalar* __restrict__ t_in,
                                           int ncam, Scalar factor, Scalar lam,
                                           Scalar* __restrict__ R_out,
                                           Scalar* __restrict__ t_out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  Scalar H[6][6];
#pragma unroll
  for (int i = 0; i < 6; ++i)
#pragma unroll
    for (int j = 0; j < 6; ++j) H[i][j] = factor * Hc_raw[36 * c + 6 * i + j];
#pragma unroll
  for (int i = 0; i < 6; ++i) H[i][i] += lam;
  Scalar b[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) b[i] = gc[6 * c + i];
  Scalar x[6];
  CholeskySolve<6>(H, b, x);
  Scalar dC[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) dC[i] = -x[i];

  Scalar dR[9];
  ExpSO3(dC, dR);
  Mat3Mul(dR, R_in + 9 * c, R_out + 9 * c);
  t_out[3 * c] = t_in[3 * c] + dC[3];
  t_out[3 * c + 1] = t_in[3 * c + 1] + dC[4];
  t_out[3 * c + 2] = t_in[3 * c + 2] + dC[5];
}

__global__ void KernelSolveRetractPoints(const Scalar* __restrict__ Hp_raw,
                                          const Scalar* __restrict__ gp,
                                          const Scalar* __restrict__ X_in, int npt,
                                          Scalar factor, Scalar lam,
                                          Scalar* __restrict__ X_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npt) return;
  Scalar H[3][3];
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) H[i][j] = factor * Hp_raw[9 * p + 3 * i + j];
#pragma unroll
  for (int i = 0; i < 3; ++i) H[i][i] += lam;
  Scalar b[3] = {gp[3 * p], gp[3 * p + 1], gp[3 * p + 2]};
  Scalar x[3];
  CholeskySolve<3>(H, b, x);
  X_out[3 * p] = X_in[3 * p] - x[0];
  X_out[3 * p + 1] = X_in[3 * p + 1] - x[1];
  X_out[3 * p + 2] = X_in[3 * p + 2] - x[2];
}

// ============================================================== Kernel D: Nesterov extrapolation
__global__ void KernelExtrapolateCameras(const Scalar* __restrict__ R_curr,
                                          const Scalar* __restrict__ R_prev,
                                          const Scalar* __restrict__ t_curr,
                                          const Scalar* __restrict__ t_prev, int ncam,
                                          Scalar beta, Scalar* __restrict__ R_y,
                                          Scalar* __restrict__ t_y) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  Scalar RprevT[9];
  const Scalar* Rp = R_prev + 9 * c;
  // R_prev^T
  RprevT[0] = Rp[0]; RprevT[1] = Rp[3]; RprevT[2] = Rp[6];
  RprevT[3] = Rp[1]; RprevT[4] = Rp[4]; RprevT[5] = Rp[7];
  RprevT[6] = Rp[2]; RprevT[7] = Rp[5]; RprevT[8] = Rp[8];
  Scalar RR[9];
  Mat3Mul(R_curr + 9 * c, RprevT, RR);  // R_curr @ R_prev^T
  Scalar omega[3];
  LogSO3(RR, omega);
  Scalar bomega[3] = {beta * omega[0], beta * omega[1], beta * omega[2]};
  Scalar dR[9];
  ExpSO3(bomega, dR);
  Mat3Mul(dR, R_curr + 9 * c, R_y + 9 * c);
  for (int d = 0; d < 3; ++d) {
    t_y[3 * c + d] = t_curr[3 * c + d] + beta * (t_curr[3 * c + d] - t_prev[3 * c + d]);
  }
}
__global__ void KernelExtrapolatePoints(const Scalar* __restrict__ X_curr,
                                         const Scalar* __restrict__ X_prev, int npt,
                                         Scalar beta, Scalar* __restrict__ X_y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * npt) return;
  X_y[i] = X_curr[i] + beta * (X_curr[i] - X_prev[i]);
}

// ============================================================== host orchestration
struct DeviceProblem {
  int ncam, npt, nobs;
  int *cam_idx, *pt_idx;
  Scalar *uv, *f, *k1, *k2;
};

struct DeviceState {
  Scalar *R, *t, *X;
};

DeviceState AllocState(int ncam, int npt) {
  DeviceState s;
  CUDA_CHECK(cudaMalloc(&s.R, 9 * ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&s.t, 3 * ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&s.X, 3 * npt * sizeof(Scalar)));
  return s;
}
void CopyState(const DeviceState& dst, const DeviceState& src, int ncam, int npt) {
  CUDA_CHECK(cudaMemcpy(dst.R, src.R, 9 * ncam * sizeof(Scalar), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(dst.t, src.t, 3 * ncam * sizeof(Scalar), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(dst.X, src.X, 3 * npt * sizeof(Scalar), cudaMemcpyDeviceToDevice));
}

inline int GridSize(int n, int block) { return (n + block - 1) / block; }

Scalar ComputeCost(const DeviceProblem& p, const DeviceState& s, Scalar* d_cost) {
  Scalar zero = 0;
  CUDA_CHECK(cudaMemcpy(d_cost, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  const int block = 256;
  KernelCostOnly<<<GridSize(p.nobs, block), block>>>(p.cam_idx, p.pt_idx, p.uv, s.R, s.t,
                                                       s.X, p.f, p.k1, p.k2, p.nobs, d_cost);
  Scalar cost;
  CUDA_CHECK(cudaMemcpy(&cost, d_cost, sizeof(Scalar), cudaMemcpyDeviceToHost));
  return cost;
}

// One MM step: linearize at `y`, solve, retract into `out`. Returns cost(out).
Scalar MmStep(const DeviceProblem& p, const DeviceState& y, const DeviceState& out,
              Scalar* Hc, Scalar* gc, Scalar* Hp, Scalar* gp, Scalar* d_cost,
              Scalar factor, Scalar lam) {
  CUDA_CHECK(cudaMemset(Hc, 0, (size_t)36 * p.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(gc, 0, (size_t)6 * p.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(Hp, 0, (size_t)9 * p.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(gp, 0, (size_t)3 * p.npt * sizeof(Scalar)));
  Scalar zero = 0;
  CUDA_CHECK(cudaMemcpy(d_cost, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));

  const int block = 256;
  KernelJacobianAccumulate<<<GridSize(p.nobs, block), block>>>(
      p.cam_idx, p.pt_idx, p.uv, y.R, y.t, y.X, p.f, p.k1, p.k2, p.nobs, Hc, gc, Hp, gp,
      d_cost);
  CUDA_CHECK(cudaGetLastError());

  KernelSolveRetractCameras<<<GridSize(p.ncam, block), block>>>(Hc, gc, y.R, y.t, p.ncam,
                                                                 factor, lam, out.R, out.t);
  KernelSolveRetractPoints<<<GridSize(p.npt, block), block>>>(Hp, gp, y.X, p.npt, factor,
                                                                lam, out.X);
  CUDA_CHECK(cudaGetLastError());
  return ComputeCost(p, out, d_cost);
}

namespace {
void PrintUsage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s --dataset <bal.txt> [--iters N] [--accelerated 0|1]\n"
               "  [--factor F] [--lam L] [--eta E] [--loss l2|huber] [--fp64 0|1]\n"
               "\n"
               "  --dataset      path to a BAL problem file (required)\n"
               "  --iters        outer MM iterations (default 200)\n"
               "  --accelerated  1 = Nesterov + restart (default), 0 = plain MM\n"
               "  --factor       majorization factor (default 2.0, per the spec)\n"
               "  --lam          block-solve Levenberg damping (default 1e-6)\n"
               "  --eta          EMA weight for the restart reference cost (default\n"
               "                 1.0 = EMA reduces to the last cost, i.e. restart\n"
               "                 trigger matches the original hard-restart rule;\n"
               "                 only the momentum counter q is still halved, not\n"
               "                 reset, on restart). Fan et al. (arXiv:2108.00083\n"
               "                 Remark 10) recommend eta << 1 on distributed pose-\n"
               "                 graph MM and report it reduces unneeded restarts;\n"
               "                 measured directly on this repo's real BA problem,\n"
               "                 eta <= ~0.2 instead makes the 200-it final cost\n"
               "                 ~0.6%% WORSE (6.8955e4 vs 6.8531e4 baseline), eta\n"
               "                 >= ~0.25 is a statistical wash. Their finding does\n"
               "                 not transfer to this problem class -- kept as a\n"
               "                 tunable for further experimentation, not the\n"
               "                 default. See CONVERGENCE_LITERATURE.md.\n"
               "  --loss         l2 (default) or huber; huber is not implemented\n"
               "                 (optional extension, Section 7) -- passing it\n"
               "                 prints a warning and falls back to l2 rather\n"
               "                 than silently doing the wrong thing.\n"
               "  --fp64         1 (default, only supported value); fp32 is not\n"
               "                 implemented (Scalar is a single typedef so it's\n"
               "                 a one-line change, but that path is untested;\n"
               "                 see README).\n",
               prog);
}
}  // namespace

int main(int argc, char** argv) {
  std::string path;
  int n_iter = 200;
  bool accelerated = true;
  Scalar factor = 2.0;
  Scalar lam = 1e-6;
  Scalar eta = 1.0;
  std::string loss = "l2";
  bool fp64 = true;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (a == "--dataset") path = next();
    else if (a == "--iters") n_iter = std::atoi(next().c_str());
    else if (a == "--accelerated") accelerated = std::atoi(next().c_str()) != 0;
    else if (a == "--factor") factor = std::atof(next().c_str());
    else if (a == "--lam") lam = std::atof(next().c_str());
    else if (a == "--eta") eta = std::atof(next().c_str());
    else if (a == "--loss") loss = next();
    else if (a == "--fp64") fp64 = std::atoi(next().c_str()) != 0;
    else if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return EXIT_SUCCESS; }
    else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); PrintUsage(argv[0]); return EXIT_FAILURE; }
  }
  if (path.empty()) { PrintUsage(argv[0]); return EXIT_FAILURE; }
  if (loss != "l2") {
    std::fprintf(stderr,
                 "warning: --loss %s not implemented (optional extension, spec "
                 "Section 7) -- falling back to l2\n",
                 loss.c_str());
  }
  if (!fp64) {
    std::fprintf(stderr, "warning: --fp64 0 (fp32) not implemented -- running fp64\n");
  }

  BalData bal = LoadBal(path);
  std::printf("loaded %s: ncam=%d npt=%d nobs=%d\n", path.c_str(), bal.ncam, bal.npt,
              bal.nobs);

  // host-side R0 from angle-axis (Exp), matching reference_mm.py's rotvec_to_R(cams[:,0:3])
  std::vector<Scalar> R0(9 * bal.ncam), t0(3 * bal.ncam), f0(bal.ncam), k10(bal.ncam),
      k20(bal.ncam);
  for (int c = 0; c < bal.ncam; ++c) {
    Scalar a[3] = {bal.cams[9 * c], bal.cams[9 * c + 1], bal.cams[9 * c + 2]};
    Scalar th = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    Scalar R[9];
    if (th < 1e-12) {
      R[0] = 1; R[1] = 0; R[2] = 0; R[3] = 0; R[4] = 1; R[5] = 0; R[6] = 0; R[7] = 0; R[8] = 1;
    } else {
      Scalar kx = a[0] / th, ky = a[1] / th, kz = a[2] / th;
      Scalar c_ = std::cos(th), s_ = std::sin(th), C = 1 - c_;
      R[0] = c_ + kx * kx * C;      R[1] = kx * ky * C - kz * s_; R[2] = kx * kz * C + ky * s_;
      R[3] = ky * kx * C + kz * s_; R[4] = c_ + ky * ky * C;      R[5] = ky * kz * C - kx * s_;
      R[6] = kz * kx * C - ky * s_; R[7] = kz * ky * C + kx * s_; R[8] = c_ + kz * kz * C;
    }
    for (int i = 0; i < 9; ++i) R0[9 * c + i] = R[i];
    t0[3 * c] = bal.cams[9 * c + 3]; t0[3 * c + 1] = bal.cams[9 * c + 4];
    t0[3 * c + 2] = bal.cams[9 * c + 5];
    f0[c] = bal.cams[9 * c + 6]; k10[c] = bal.cams[9 * c + 7]; k20[c] = bal.cams[9 * c + 8];
  }

  DeviceProblem p;
  p.ncam = bal.ncam; p.npt = bal.npt; p.nobs = bal.nobs;
  CUDA_CHECK(cudaMalloc(&p.cam_idx, bal.nobs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&p.pt_idx, bal.nobs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&p.uv, 2 * bal.nobs * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&p.f, bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&p.k1, bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&p.k2, bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMemcpy(p.cam_idx, bal.cam_idx.data(), bal.nobs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p.pt_idx, bal.pt_idx.data(), bal.nobs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p.uv, bal.uv.data(), 2 * bal.nobs * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p.f, f0.data(), bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p.k1, k10.data(), bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(p.k2, k20.data(), bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));

  DeviceState curr = AllocState(bal.ncam, bal.npt);
  DeviceState prev = AllocState(bal.ncam, bal.npt);
  DeviceState y = AllocState(bal.ncam, bal.npt);
  DeviceState next = AllocState(bal.ncam, bal.npt);
  CUDA_CHECK(cudaMemcpy(curr.R, R0.data(), 9 * bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(curr.t, t0.data(), 3 * bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(curr.X, bal.pts.data(), 3 * bal.npt * sizeof(Scalar), cudaMemcpyHostToDevice));
  CopyState(prev, curr, bal.ncam, bal.npt);

  Scalar *Hc, *gc, *Hp, *gp, *d_cost;
  CUDA_CHECK(cudaMalloc(&Hc, (size_t)36 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&gc, (size_t)6 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&Hp, (size_t)9 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&gp, (size_t)3 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_cost, sizeof(Scalar)));

  Scalar cost_curr = ComputeCost(p, curr, d_cost);
  Scalar rmse0 = std::sqrt(cost_curr * 2.0 / bal.nobs);
  std::printf("init cost=%.6e rmse=%.4fpx\n", cost_curr, rmse0);

  const int block = 256;
  Scalar q = 1;
  // EMA restart reference (arXiv:2108.00083 Eq. 59): F_ema(0) = F(X(0)), then
  // updated after every accepted iterate. Restarting against this running average
  // instead of the single previous cost tolerates small non-monotone fluctuations
  // that would otherwise trigger a restart, which the source paper's own authors
  // found reduced unnecessary restarts and sped up convergence relative to the
  // plain "restart on any increase" rule this file used before (see
  // CONVERGENCE_LITERATURE.md). The sufficient-decrease tolerance term (psi term
  // in the paper's Algorithm 4) is deliberately not ported here -- no calibrated
  // value for it was available from the accessible parts of that paper, and an
  // uncalibrated constant would be worse than omitting it.
  Scalar cost_ema = cost_curr;
  auto t_start = std::chrono::steady_clock::now();
  for (int it = 1; it <= n_iter; ++it) {
    if (accelerated) {
      Scalar beta = (q - 1) / (q + 2);
      KernelExtrapolateCameras<<<GridSize(bal.ncam, block), block>>>(
          curr.R, prev.R, curr.t, prev.t, bal.ncam, beta, y.R, y.t);
      KernelExtrapolatePoints<<<GridSize(3 * bal.npt, block), block>>>(curr.X, prev.X,
                                                                        bal.npt, beta, y.X);
      CUDA_CHECK(cudaGetLastError());
    } else {
      CopyState(y, curr, bal.ncam, bal.npt);
    }

    Scalar cost_new = MmStep(p, y, next, Hc, gc, Hp, gp, d_cost, factor, lam);

    bool restarted = false;
    if (accelerated && cost_new > cost_ema) {
      cost_new = MmStep(p, curr, next, Hc, gc, Hp, gp, d_cost, factor, lam);
      q = std::max(q / (Scalar)2, (Scalar)1);
      restarted = true;
    } else {
      q += 1;
    }

    // prev := old curr, then curr := next. Order matters (must snapshot curr into
    // prev before overwriting curr) but no pointer-swap trickery is needed since
    // curr/prev/next are separate fixed allocations.
    CopyState(prev, curr, bal.ncam, bal.npt);
    CopyState(curr, next, bal.ncam, bal.npt);
    cost_curr = cost_new;
    cost_ema = (1 - eta) * cost_ema + eta * cost_curr;

    if (it % 10 == 0 || it == n_iter) {
      std::printf("  it%4d cost=%.5e ema=%.5e q=%.3f restart=%d\n", it, cost_curr, cost_ema, q,
                  restarted);
    }
  }
  cudaDeviceSynchronize();
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::printf("done: %d iters, wall=%.3fs (%.2f ms/iter), final cost=%.6e\n", n_iter, wall,
              1000.0 * wall / n_iter, cost_curr);
  return 0;
}
