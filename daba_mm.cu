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

// ============================================================== Kernel E: gradient-restart dot product
// grad_f(y)^T (x_new - x_curr), per O'Donoghue & Candes (arXiv:1204.3982 Sec 3.2)
// "gradient scheme": restart whenever this is > 0. gc/gp are grad_f(y) -- already
// computed by KernelJacobianAccumulate inside MmStep, not recomputed here. The
// step (x_new - x_curr) is taken in the tangent space at R_curr for rotations
// (Log(R_new @ R_curr^T)), Euclidean for translations/points.
__global__ void KernelGradientRestartDotCameras(const Scalar* __restrict__ R_new,
                                                 const Scalar* __restrict__ R_curr,
                                                 const Scalar* __restrict__ t_new,
                                                 const Scalar* __restrict__ t_curr,
                                                 const Scalar* __restrict__ gc, int ncam,
                                                 Scalar* __restrict__ dot_out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  Scalar RcurrT[9];
  const Scalar* Rc = R_curr + 9 * c;
  RcurrT[0] = Rc[0]; RcurrT[1] = Rc[3]; RcurrT[2] = Rc[6];
  RcurrT[3] = Rc[1]; RcurrT[4] = Rc[4]; RcurrT[5] = Rc[7];
  RcurrT[6] = Rc[2]; RcurrT[7] = Rc[5]; RcurrT[8] = Rc[8];
  Scalar RR[9];
  Mat3Mul(R_new + 9 * c, RcurrT, RR);  // R_new @ R_curr^T
  Scalar omega[3];
  LogSO3(RR, omega);
  Scalar dt0 = t_new[3 * c] - t_curr[3 * c];
  Scalar dt1 = t_new[3 * c + 1] - t_curr[3 * c + 1];
  Scalar dt2 = t_new[3 * c + 2] - t_curr[3 * c + 2];
  const Scalar* g = gc + 6 * c;
  Scalar dot = g[0] * omega[0] + g[1] * omega[1] + g[2] * omega[2] + g[3] * dt0 + g[4] * dt1 +
               g[5] * dt2;
  atomicAdd(dot_out, dot);
}

__global__ void KernelGradientRestartDotPoints(const Scalar* __restrict__ X_new,
                                                const Scalar* __restrict__ X_curr,
                                                const Scalar* __restrict__ gp, int npt,
                                                Scalar* __restrict__ dot_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npt) return;
  Scalar dx0 = X_new[3 * p] - X_curr[3 * p];
  Scalar dx1 = X_new[3 * p + 1] - X_curr[3 * p + 1];
  Scalar dx2 = X_new[3 * p + 2] - X_curr[3 * p + 2];
  const Scalar* g = gp + 3 * p;
  atomicAdd(dot_out, g[0] * dx0 + g[1] * dx1 + g[2] * dx2);
}

// ============================================================== Kernel F: BB secant dot products
// s.s and s.ydiff for the BB-adaptive majorization factor (see BBFactor below and
// CONVERGENCE_LITERATURE.md for the derivation/caveats). s = tangent-space step from
// y_prev to y (Log(R_y @ R_yprev^T) for rotations, Euclidean for t/X); ydiff = grad_f(y)
// - grad_f(y_prev), i.e. (gc,gp) - (gc_prev,gp_prev), both already computed by
// AccumulateGradHess -- no extra Jacobian pass.
__global__ void KernelBBDotsCameras(const Scalar* __restrict__ R_y,
                                     const Scalar* __restrict__ R_yprev,
                                     const Scalar* __restrict__ t_y,
                                     const Scalar* __restrict__ t_yprev,
                                     const Scalar* __restrict__ gc,
                                     const Scalar* __restrict__ gc_prev, int ncam,
                                     Scalar* __restrict__ ss_out, Scalar* __restrict__ syd_out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  Scalar RyprevT[9];
  const Scalar* Rp = R_yprev + 9 * c;
  RyprevT[0] = Rp[0]; RyprevT[1] = Rp[3]; RyprevT[2] = Rp[6];
  RyprevT[3] = Rp[1]; RyprevT[4] = Rp[4]; RyprevT[5] = Rp[7];
  RyprevT[6] = Rp[2]; RyprevT[7] = Rp[5]; RyprevT[8] = Rp[8];
  Scalar RR[9];
  Mat3Mul(R_y + 9 * c, RyprevT, RR);  // R_y @ R_yprev^T
  Scalar somega[3];
  LogSO3(RR, somega);
  Scalar s[6] = {somega[0], somega[1], somega[2], t_y[3 * c] - t_yprev[3 * c],
                 t_y[3 * c + 1] - t_yprev[3 * c + 1], t_y[3 * c + 2] - t_yprev[3 * c + 2]};
  const Scalar* g = gc + 6 * c;
  const Scalar* gp0 = gc_prev + 6 * c;
  Scalar ss = 0, syd = 0;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    ss += s[i] * s[i];
    syd += s[i] * (g[i] - gp0[i]);
  }
  atomicAdd(ss_out, ss);
  atomicAdd(syd_out, syd);
}

__global__ void KernelBBDotsPoints(const Scalar* __restrict__ X_y,
                                    const Scalar* __restrict__ X_yprev,
                                    const Scalar* __restrict__ gp,
                                    const Scalar* __restrict__ gp_prev, int npt,
                                    Scalar* __restrict__ ss_out, Scalar* __restrict__ syd_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npt) return;
  Scalar s[3] = {X_y[3 * p] - X_yprev[3 * p], X_y[3 * p + 1] - X_yprev[3 * p + 1],
                 X_y[3 * p + 2] - X_yprev[3 * p + 2]};
  const Scalar* g = gp + 3 * p;
  const Scalar* g0 = gp_prev + 3 * p;
  Scalar ss = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
  Scalar syd = s[0] * (g[0] - g0[0]) + s[1] * (g[1] - g0[1]) + s[2] * (g[2] - g0[2]);
  atomicAdd(ss_out, ss);
  atomicAdd(syd_out, syd);
}

// ============================================================== Kernel G: SQUAREM r, v, norms
// r = tangent diff X1-X0, s2 = tangent diff X2-X1, v = s2-r; also accumulates ||r||^2,
// ||v||^2 (S3 scheme -- see solve_squarem in reference_mm.py for the full derivation,
// caveats, and the mm_step-can-increase-cost finding this was built to chase down).
__global__ void KernelSquaremRVCameras(const Scalar* __restrict__ R0, const Scalar* __restrict__ R1,
                                        const Scalar* __restrict__ R2, const Scalar* __restrict__ t0,
                                        const Scalar* __restrict__ t1, const Scalar* __restrict__ t2,
                                        int ncam, Scalar* __restrict__ r_out, Scalar* __restrict__ v_out,
                                        Scalar* __restrict__ rnorm2_out, Scalar* __restrict__ vnorm2_out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  Scalar R0T[9];
  const Scalar* Rp0 = R0 + 9 * c;
  R0T[0] = Rp0[0]; R0T[1] = Rp0[3]; R0T[2] = Rp0[6];
  R0T[3] = Rp0[1]; R0T[4] = Rp0[4]; R0T[5] = Rp0[7];
  R0T[6] = Rp0[2]; R0T[7] = Rp0[5]; R0T[8] = Rp0[8];
  Scalar RR1[9];
  Mat3Mul(R1 + 9 * c, R0T, RR1);  // R1 @ R0^T
  Scalar r_omega[3];
  LogSO3(RR1, r_omega);

  Scalar R1T[9];
  const Scalar* Rp1 = R1 + 9 * c;
  R1T[0] = Rp1[0]; R1T[1] = Rp1[3]; R1T[2] = Rp1[6];
  R1T[3] = Rp1[1]; R1T[4] = Rp1[4]; R1T[5] = Rp1[7];
  R1T[6] = Rp1[2]; R1T[7] = Rp1[5]; R1T[8] = Rp1[8];
  Scalar RR2[9];
  Mat3Mul(R2 + 9 * c, R1T, RR2);  // R2 @ R1^T
  Scalar s2_omega[3];
  LogSO3(RR2, s2_omega);

  Scalar r6[6] = {r_omega[0], r_omega[1], r_omega[2], t1[3 * c] - t0[3 * c],
                   t1[3 * c + 1] - t0[3 * c + 1], t1[3 * c + 2] - t0[3 * c + 2]};
  Scalar s2_6[6] = {s2_omega[0], s2_omega[1], s2_omega[2], t2[3 * c] - t1[3 * c],
                     t2[3 * c + 1] - t1[3 * c + 1], t2[3 * c + 2] - t1[3 * c + 2]};
  Scalar* ro = r_out + 6 * c;
  Scalar* vo = v_out + 6 * c;
  Scalar rn2 = 0, vn2 = 0;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    ro[i] = r6[i];
    vo[i] = s2_6[i] - r6[i];
    rn2 += ro[i] * ro[i];
    vn2 += vo[i] * vo[i];
  }
  atomicAdd(rnorm2_out, rn2);
  atomicAdd(vnorm2_out, vn2);
}

__global__ void KernelSquaremRVPoints(const Scalar* __restrict__ X0, const Scalar* __restrict__ X1,
                                       const Scalar* __restrict__ X2, int npt,
                                       Scalar* __restrict__ r_out, Scalar* __restrict__ v_out,
                                       Scalar* __restrict__ rnorm2_out, Scalar* __restrict__ vnorm2_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npt) return;
  Scalar rn2 = 0, vn2 = 0;
  Scalar* ro = r_out + 3 * p;
  Scalar* vo = v_out + 3 * p;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    Scalar r = X1[3 * p + i] - X0[3 * p + i];
    Scalar s2 = X2[3 * p + i] - X1[3 * p + i];
    Scalar v = s2 - r;
    ro[i] = r;
    vo[i] = v;
    rn2 += r * r;
    vn2 += v * v;
  }
  atomicAdd(rnorm2_out, rn2);
  atomicAdd(vnorm2_out, vn2);
}

// Kernel H: retract X0 by (-2*alpha*r + alpha^2*v), the S3 SQUAREM candidate point.
__global__ void KernelSquaremRetractCameras(const Scalar* __restrict__ R0, const Scalar* __restrict__ t0,
                                             const Scalar* __restrict__ r, const Scalar* __restrict__ v,
                                             Scalar alpha, int ncam, Scalar* __restrict__ R_out,
                                             Scalar* __restrict__ t_out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  const Scalar* rc = r + 6 * c;
  const Scalar* vc = v + 6 * c;
  Scalar upd[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) upd[i] = -2 * alpha * rc[i] + alpha * alpha * vc[i];
  Scalar dR[9];
  ExpSO3(upd, dR);
  Mat3Mul(dR, R0 + 9 * c, R_out + 9 * c);
  t_out[3 * c] = t0[3 * c] + upd[3];
  t_out[3 * c + 1] = t0[3 * c + 1] + upd[4];
  t_out[3 * c + 2] = t0[3 * c + 2] + upd[5];
}

__global__ void KernelSquaremRetractPoints(const Scalar* __restrict__ X0, const Scalar* __restrict__ r,
                                            const Scalar* __restrict__ v, Scalar alpha, int npt,
                                            Scalar* __restrict__ X_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npt) return;
  const Scalar* rp = r + 3 * p;
  const Scalar* vp = v + 3 * p;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    X_out[3 * p + i] = X0[3 * p + i] + (-2 * alpha * rp[i] + alpha * alpha * vp[i]);
  }
}

// ============================================================== Kernel H: multi-lambda per-block damping
// Adapted from https://github.com/msouiai/multishift-bundle-adjustment: that repo solves
// the FULL coupled normal equations with CG, exploiting Krylov shift-invariance so one
// matvec stream serves a whole grid of LM lambda values, picks whichever candidate has the
// lowest TRUE (non-linearized) cost, greedy min-cost beating gain-ratio selection by their
// own finding. Doesn't transplant literally -- DABA's blocks are independent 6x6/3x3 dense
// systems already solved by direct Cholesky, no large system to amortize a Krylov trick
// over. What transplants is the philosophy: try a small lambda grid per block (cheap --
// n_lambda extra tiny Cholesky solves next to the O(nobs) accumulate pass that actually
// dominates wall-clock), evaluate each by TRUE local nonlinear cost, keep the best if it
// improves. Adaptation: PER-BLOCK accept/reject, not one global lambda -- an ill-
// conditioned block freezes at y instead of moving by a lambda tuned for the average block
// or corrupting neighbors next iteration. See reference_mm.py's solve_retract_multilambda
// for the derivation and CONVERGENCE_LITERATURE.md for why this matters (the insta360x4
// NaN-cascade failure this targets).

// Per-block cost at the CURRENT (pre-retraction, linearization-point) state -- the
// baseline every candidate must beat. Separate pass from KernelJacobianAccumulate (not
// fused into it) to avoid touching that already-validated kernel; the extra O(nobs) pass
// is cheap next to the Jacobian accumulation work already paid for this iteration.
__global__ void KernelPerBlockCostNow(const int* __restrict__ cam_idx,
                                       const int* __restrict__ pt_idx,
                                       const Scalar* __restrict__ uv,
                                       const Scalar* __restrict__ R, const Scalar* __restrict__ t,
                                       const Scalar* __restrict__ X, const Scalar* __restrict__ f,
                                       const Scalar* __restrict__ k1, const Scalar* __restrict__ k2,
                                       int nobs, Scalar* __restrict__ cam_cost_now,
                                       Scalar* __restrict__ pt_cost_now) {
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
  Scalar cost = 0.5 * (res0 * res0 + res1 * res1);
  atomicAdd(&cam_cost_now[c], cost);
  atomicAdd(&pt_cost_now[p], cost);
}

// One thread per (camera, candidate). lam_mult: small [n_lambda] grid of relative
// multipliers, shared by every block; base_lam_cam: per-camera persistent state.
__global__ void KernelMultiLambdaSolveCameras(
    const Scalar* __restrict__ Hc_raw, const Scalar* __restrict__ gc,
    const Scalar* __restrict__ base_lam_cam, const Scalar* __restrict__ lam_mult,
    int ncam, int n_lambda, Scalar factor, Scalar* __restrict__ dC_cand,
    Scalar* __restrict__ lam_used) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = ncam * n_lambda;
  if (idx >= total) return;
  int c = idx / n_lambda, k = idx % n_lambda;
  Scalar lam = base_lam_cam[c] * lam_mult[k];
  lam_used[idx] = lam;
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
#pragma unroll
  for (int i = 0; i < 6; ++i) dC_cand[6 * idx + i] = -x[i];
}

__global__ void KernelMultiLambdaSolvePoints(
    const Scalar* __restrict__ Hp_raw, const Scalar* __restrict__ gp,
    const Scalar* __restrict__ base_lam_pt, const Scalar* __restrict__ lam_mult,
    int npt, int n_lambda, Scalar factor, Scalar* __restrict__ dX_cand,
    Scalar* __restrict__ lam_used) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = npt * n_lambda;
  if (idx >= total) return;
  int p = idx / n_lambda, k = idx % n_lambda;
  Scalar lam = base_lam_pt[p] * lam_mult[k];
  lam_used[idx] = lam;
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
#pragma unroll
  for (int i = 0; i < 3; ++i) dX_cand[3 * idx + i] = -x[i];
}

// One thread per (observation, candidate): evaluate TRUE local cost of each camera
// candidate step, holding points fixed at the CURRENT y-state (same linearization-point
// assumption Hc/gc were computed under). dC_cand indexed [c*n_lambda+k].
__global__ void KernelMultiLambdaCandidateCostCameras(
    const int* __restrict__ cam_idx, const int* __restrict__ pt_idx,
    const Scalar* __restrict__ uv, const Scalar* __restrict__ R, const Scalar* __restrict__ t,
    const Scalar* __restrict__ X, const Scalar* __restrict__ f, const Scalar* __restrict__ k1,
    const Scalar* __restrict__ k2, const Scalar* __restrict__ dC_cand, int nobs, int n_lambda,
    Scalar* __restrict__ cam_cost_cand) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nobs * n_lambda;
  if (idx >= total) return;
  int o = idx / n_lambda, k = idx % n_lambda;
  int c = cam_idx[o], p = pt_idx[o];
  const Scalar* dC = dC_cand + 6 * (c * n_lambda + k);
  Scalar dR[9];
  ExpSO3(dC, dR);
  Scalar Rc[9];
  Mat3Mul(dR, R + 9 * c, Rc);
  Scalar tx = t[3 * c] + dC[3], ty = t[3 * c + 1] + dC[4], tz = t[3 * c + 2] + dC[5];

  Scalar Xp0 = X[3 * p], Xp1 = X[3 * p + 1], Xp2 = X[3 * p + 2];
  Scalar rx = Rc[0] * Xp0 + Rc[1] * Xp1 + Rc[2] * Xp2;
  Scalar ry = Rc[3] * Xp0 + Rc[4] * Xp1 + Rc[5] * Xp2;
  Scalar rz = Rc[6] * Xp0 + Rc[7] * Xp1 + Rc[8] * Xp2;
  Scalar Px = rx + tx, Py = ry + ty, Pz = rz + tz;
  Scalar xp = -Px / Pz, yp = -Py / Pz;
  Scalar r2 = xp * xp + yp * yp;
  Scalar fo = f[c], k1o = k1[c], k2o = k2[c];
  Scalar dist = 1.0 + k1o * r2 + k2o * r2 * r2;
  Scalar res0 = fo * dist * xp - uv[2 * o];
  Scalar res1 = fo * dist * yp - uv[2 * o + 1];
  atomicAdd(&cam_cost_cand[c * n_lambda + k], 0.5 * (res0 * res0 + res1 * res1));
}

// Symmetric: point candidates, holding camera poses fixed at y.
__global__ void KernelMultiLambdaCandidateCostPoints(
    const int* __restrict__ cam_idx, const int* __restrict__ pt_idx,
    const Scalar* __restrict__ uv, const Scalar* __restrict__ R, const Scalar* __restrict__ t,
    const Scalar* __restrict__ X, const Scalar* __restrict__ f, const Scalar* __restrict__ k1,
    const Scalar* __restrict__ k2, const Scalar* __restrict__ dX_cand, int nobs, int n_lambda,
    Scalar* __restrict__ pt_cost_cand) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nobs * n_lambda;
  if (idx >= total) return;
  int o = idx / n_lambda, k = idx % n_lambda;
  int c = cam_idx[o], p = pt_idx[o];
  const Scalar* dX = dX_cand + 3 * (p * n_lambda + k);
  Scalar Xp0 = X[3 * p] + dX[0], Xp1 = X[3 * p + 1] + dX[1], Xp2 = X[3 * p + 2] + dX[2];

  const Scalar* Rc = R + 9 * c;
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
  atomicAdd(&pt_cost_cand[p * n_lambda + k], 0.5 * (res0 * res0 + res1 * res1));
}

// One thread per camera: pick the best candidate (if any beats cam_cost_now), retract
// into R_new/t_new, update base_lam_cam for next iteration (accept: chosen/3; reject:
// base_lam_cam*10), count accepted blocks for logging.
__global__ void KernelMultiLambdaSelectCameras(
    const Scalar* __restrict__ R, const Scalar* __restrict__ t,
    const Scalar* __restrict__ dC_cand, const Scalar* __restrict__ lam_used,
    const Scalar* __restrict__ cam_cost_cand, const Scalar* __restrict__ cam_cost_now,
    int ncam, int n_lambda, Scalar lam_floor, Scalar lam_ceil,
    Scalar* __restrict__ R_new, Scalar* __restrict__ t_new,
    Scalar* __restrict__ base_lam_cam, int* __restrict__ num_accepted) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncam) return;
  int best_k = 0;
  Scalar best_cost = cam_cost_cand[c * n_lambda];
#pragma unroll
  for (int k = 1; k < n_lambda; ++k) {
    Scalar ck = cam_cost_cand[c * n_lambda + k];
    if (ck < best_cost) { best_cost = ck; best_k = k; }
  }
  if (best_cost < cam_cost_now[c]) {
    const Scalar* dC = dC_cand + 6 * (c * n_lambda + best_k);
    Scalar dR[9];
    ExpSO3(dC, dR);
    Mat3Mul(dR, R + 9 * c, R_new + 9 * c);
    t_new[3 * c] = t[3 * c] + dC[3];
    t_new[3 * c + 1] = t[3 * c + 1] + dC[4];
    t_new[3 * c + 2] = t[3 * c + 2] + dC[5];
    Scalar target = lam_used[c * n_lambda + best_k] / (Scalar)3;
    base_lam_cam[c] = target < lam_floor ? lam_floor : (target > lam_ceil ? lam_ceil : target);
    atomicAdd(num_accepted, 1);
  } else {
    for (int i = 0; i < 9; ++i) R_new[9 * c + i] = R[9 * c + i];
    t_new[3 * c] = t[3 * c]; t_new[3 * c + 1] = t[3 * c + 1]; t_new[3 * c + 2] = t[3 * c + 2];
    Scalar target = base_lam_cam[c] * (Scalar)10;
    base_lam_cam[c] = target < lam_floor ? lam_floor : (target > lam_ceil ? lam_ceil : target);
  }
}

__global__ void KernelMultiLambdaSelectPoints(
    const Scalar* __restrict__ X, const Scalar* __restrict__ dX_cand,
    const Scalar* __restrict__ lam_used, const Scalar* __restrict__ pt_cost_cand,
    const Scalar* __restrict__ pt_cost_now, int npt, int n_lambda, Scalar lam_floor,
    Scalar lam_ceil, Scalar* __restrict__ X_new, Scalar* __restrict__ base_lam_pt,
    int* __restrict__ num_accepted) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npt) return;
  int best_k = 0;
  Scalar best_cost = pt_cost_cand[p * n_lambda];
#pragma unroll
  for (int k = 1; k < n_lambda; ++k) {
    Scalar ck = pt_cost_cand[p * n_lambda + k];
    if (ck < best_cost) { best_cost = ck; best_k = k; }
  }
  if (best_cost < pt_cost_now[p]) {
    const Scalar* dX = dX_cand + 3 * (p * n_lambda + best_k);
    X_new[3 * p] = X[3 * p] + dX[0];
    X_new[3 * p + 1] = X[3 * p + 1] + dX[1];
    X_new[3 * p + 2] = X[3 * p + 2] + dX[2];
    Scalar target = lam_used[p * n_lambda + best_k] / (Scalar)3;
    base_lam_pt[p] = target < lam_floor ? lam_floor : (target > lam_ceil ? lam_ceil : target);
    atomicAdd(num_accepted, 1);
  } else {
    X_new[3 * p] = X[3 * p]; X_new[3 * p + 1] = X[3 * p + 1]; X_new[3 * p + 2] = X[3 * p + 2];
    Scalar target = base_lam_pt[p] * (Scalar)10;
    base_lam_pt[p] = target < lam_floor ? lam_floor : (target > lam_ceil ? lam_ceil : target);
  }
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
// Split into accumulate (build J^T J, J^T r at y -- independent of factor) and
// solve_retract (apply factor/lam, solve blocks, retract), so a factor can be chosen
// *after* seeing the gradient at y but *before* solving -- needed by the BB adaptive
// factor scheme below. Mirrors reference_mm.py's accumulate_grad_hess/solve_retract split.
void AccumulateGradHess(const DeviceProblem& p, const DeviceState& y, Scalar* Hc, Scalar* gc,
                         Scalar* Hp, Scalar* gp, Scalar* d_cost_scratch) {
  CUDA_CHECK(cudaMemset(Hc, 0, (size_t)36 * p.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(gc, 0, (size_t)6 * p.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(Hp, 0, (size_t)9 * p.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(gp, 0, (size_t)3 * p.npt * sizeof(Scalar)));
  Scalar zero = 0;
  CUDA_CHECK(cudaMemcpy(d_cost_scratch, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  const int block = 256;
  KernelJacobianAccumulate<<<GridSize(p.nobs, block), block>>>(
      p.cam_idx, p.pt_idx, p.uv, y.R, y.t, y.X, p.f, p.k1, p.k2, p.nobs, Hc, gc, Hp, gp,
      d_cost_scratch);
  CUDA_CHECK(cudaGetLastError());
}

void SolveRetract(const DeviceProblem& p, const DeviceState& y, const DeviceState& out,
                   Scalar* Hc, Scalar* gc, Scalar* Hp, Scalar* gp, Scalar factor, Scalar lam) {
  const int block = 256;
  KernelSolveRetractCameras<<<GridSize(p.ncam, block), block>>>(Hc, gc, y.R, y.t, p.ncam,
                                                                 factor, lam, out.R, out.t);
  KernelSolveRetractPoints<<<GridSize(p.npt, block), block>>>(Hp, gp, y.X, p.npt, factor,
                                                                lam, out.X);
  CUDA_CHECK(cudaGetLastError());
}

Scalar MmStep(const DeviceProblem& p, const DeviceState& y, const DeviceState& out,
              Scalar* Hc, Scalar* gc, Scalar* Hp, Scalar* gp, Scalar* d_cost,
              Scalar factor, Scalar lam) {
  AccumulateGradHess(p, y, Hc, gc, Hp, gp, d_cost);
  SolveRetract(p, y, out, Hc, gc, Hp, gp, factor, lam);
  return ComputeCost(p, out, d_cost);
}

// grad_f(y)^T (new - curr), gradient-restart trigger test (see Kernel E above). gc/gp
// must still hold grad_f(y) from the MmStep call that produced `new_state`.
Scalar GradientRestartDot(const DeviceState& new_state, const DeviceState& curr, Scalar* gc,
                           Scalar* gp, int ncam, int npt, Scalar* d_dot) {
  Scalar zero = 0;
  CUDA_CHECK(cudaMemcpy(d_dot, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  const int block = 256;
  KernelGradientRestartDotCameras<<<GridSize(ncam, block), block>>>(
      new_state.R, curr.R, new_state.t, curr.t, gc, ncam, d_dot);
  KernelGradientRestartDotPoints<<<GridSize(npt, block), block>>>(new_state.X, curr.X, gp, npt,
                                                                   d_dot);
  CUDA_CHECK(cudaGetLastError());
  Scalar dot;
  CUDA_CHECK(cudaMemcpy(&dot, d_dot, sizeof(Scalar), cudaMemcpyDeviceToHost));
  return dot;
}

// s.s and s.ydiff for the BB-adaptive factor (see Kernel F above).
void BBDots(const DeviceState& y, const DeviceState& y_prev, Scalar* gc, Scalar* gc_prev,
            Scalar* gp, Scalar* gp_prev, int ncam, int npt, Scalar* d_ss, Scalar* d_syd,
            Scalar* ss_out, Scalar* syd_out) {
  Scalar zero = 0;
  CUDA_CHECK(cudaMemcpy(d_ss, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_syd, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  const int block = 256;
  KernelBBDotsCameras<<<GridSize(ncam, block), block>>>(y.R, y_prev.R, y.t, y_prev.t, gc,
                                                         gc_prev, ncam, d_ss, d_syd);
  KernelBBDotsPoints<<<GridSize(npt, block), block>>>(y.X, y_prev.X, gp, gp_prev, npt, d_ss,
                                                       d_syd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(ss_out, d_ss, sizeof(Scalar), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(syd_out, d_syd, sizeof(Scalar), cudaMemcpyDeviceToHost));
}

// Our own adaptation of the Barzilai-Borwein secant principle to this code's `factor`
// knob -- see reference_mm.py's bb_factor_from_dots for the full derivation/caveats
// (mirrored exactly here). tau<1 exponentially smooths the clamped target instead of
// jumping straight to it; tau=1 is the raw/noisy textbook form.
Scalar BBFactor(Scalar s_dot_s, Scalar s_dot_yd, Scalar base_factor, Scalar prev_factor,
                 Scalar tau, Scalar min_mult = (Scalar)0.5, Scalar max_mult = (Scalar)25.0) {
  if (s_dot_s < (Scalar)1e-30) return prev_factor;
  Scalar raw = s_dot_yd / s_dot_s;
  if (!std::isfinite(raw) || raw <= 0) return prev_factor;
  Scalar target = std::max(min_mult * base_factor, std::min(max_mult * base_factor, raw));
  return (1 - tau) * prev_factor + tau * target;
}

// Per-block multi-lambda damped solve (Kernel H set above): replaces SolveRetract with
// a small lambda grid per block, true-cost selection, per-block accept/reject.
// base_lam_cam/base_lam_pt persist across outer iterations (mutated in place); d_lam_mult
// holds the (small, fixed) relative grid, uploaded once by the caller. Writes accepted
// states into out; returns accepted-block counts via out params for logging.
void SolveRetractMultiLambda(const DeviceProblem& p, const DeviceState& y, const DeviceState& out,
                              Scalar* Hc, Scalar* gc, Scalar* Hp, Scalar* gp,
                              Scalar* base_lam_cam, Scalar* base_lam_pt, Scalar* d_lam_mult,
                              int n_lambda, Scalar factor, Scalar lam_floor, Scalar lam_ceil,
                              Scalar* dC_cand, Scalar* dX_cand, Scalar* lam_used_cam,
                              Scalar* lam_used_pt, Scalar* cam_cost_cand, Scalar* pt_cost_cand,
                              Scalar* cam_cost_now, Scalar* pt_cost_now, int* d_num_accepted,
                              int* num_cam_accepted, int* num_pt_accepted) {
  const int block = 256;
  CUDA_CHECK(cudaMemset(cam_cost_now, 0, (size_t)p.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(pt_cost_now, 0, (size_t)p.npt * sizeof(Scalar)));
  KernelPerBlockCostNow<<<GridSize(p.nobs, block), block>>>(
      p.cam_idx, p.pt_idx, p.uv, y.R, y.t, y.X, p.f, p.k1, p.k2, p.nobs, cam_cost_now,
      pt_cost_now);

  KernelMultiLambdaSolveCameras<<<GridSize(p.ncam * n_lambda, block), block>>>(
      Hc, gc, base_lam_cam, d_lam_mult, p.ncam, n_lambda, factor, dC_cand, lam_used_cam);
  KernelMultiLambdaSolvePoints<<<GridSize(p.npt * n_lambda, block), block>>>(
      Hp, gp, base_lam_pt, d_lam_mult, p.npt, n_lambda, factor, dX_cand, lam_used_pt);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemset(cam_cost_cand, 0, (size_t)p.ncam * n_lambda * sizeof(Scalar)));
  CUDA_CHECK(cudaMemset(pt_cost_cand, 0, (size_t)p.npt * n_lambda * sizeof(Scalar)));
  KernelMultiLambdaCandidateCostCameras<<<GridSize(p.nobs * n_lambda, block), block>>>(
      p.cam_idx, p.pt_idx, p.uv, y.R, y.t, y.X, p.f, p.k1, p.k2, dC_cand, p.nobs, n_lambda,
      cam_cost_cand);
  KernelMultiLambdaCandidateCostPoints<<<GridSize(p.nobs * n_lambda, block), block>>>(
      p.cam_idx, p.pt_idx, p.uv, y.R, y.t, y.X, p.f, p.k1, p.k2, dX_cand, p.nobs, n_lambda,
      pt_cost_cand);
  CUDA_CHECK(cudaGetLastError());

  int zero = 0;
  CUDA_CHECK(cudaMemcpy(d_num_accepted, &zero, sizeof(int), cudaMemcpyHostToDevice));
  KernelMultiLambdaSelectCameras<<<GridSize(p.ncam, block), block>>>(
      y.R, y.t, dC_cand, lam_used_cam, cam_cost_cand, cam_cost_now, p.ncam, n_lambda,
      lam_floor, lam_ceil, out.R, out.t, base_lam_cam, d_num_accepted);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(num_cam_accepted, d_num_accepted, sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(d_num_accepted, &zero, sizeof(int), cudaMemcpyHostToDevice));
  KernelMultiLambdaSelectPoints<<<GridSize(p.npt, block), block>>>(
      y.X, dX_cand, lam_used_pt, pt_cost_cand, pt_cost_now, p.npt, n_lambda, lam_floor,
      lam_ceil, out.X, base_lam_pt, d_num_accepted);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(num_pt_accepted, d_num_accepted, sizeof(int), cudaMemcpyDeviceToHost));
}

// One SQUAREM meta-iteration (S3 scheme). Two-way safeguard matching the textbook
// design and this file's own Nesterov-restart branch (which also accepts whatever
// mm_step(curr) produces without comparing against the pre-restart cost) -- not a
// stricter standard than the baseline. Writes the accepted state into `out`, returns
// its cost, and reports via `accepted_extrapolation` whether the SQUAREM point beat
// the plain double-step (see reference_mm.py's solve_squarem for the strict_monotone
// diagnostic variant that isn't ported here -- numpy-only, its purpose was chasing
// down an anomaly, not something meant to ship).
Scalar SquaremStep(const DeviceProblem& p, const DeviceState& X0, const DeviceState& out,
                    Scalar* Hc, Scalar* gc, Scalar* Hp, Scalar* gp, Scalar* d_cost,
                    Scalar base_factor, Scalar lam, DeviceState& X1, DeviceState& X2,
                    DeviceState& Xsq, Scalar* r_cam, Scalar* v_cam, Scalar* r_pt, Scalar* v_pt,
                    Scalar* d_rnorm2, Scalar* d_vnorm2, bool* accepted_extrapolation) {
  MmStep(p, X0, X1, Hc, gc, Hp, gp, d_cost, base_factor, lam);
  Scalar cost2 = MmStep(p, X1, X2, Hc, gc, Hp, gp, d_cost, base_factor, lam);

  Scalar zero = 0;
  CUDA_CHECK(cudaMemcpy(d_rnorm2, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vnorm2, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
  const int block = 256;
  KernelSquaremRVCameras<<<GridSize(p.ncam, block), block>>>(
      X0.R, X1.R, X2.R, X0.t, X1.t, X2.t, p.ncam, r_cam, v_cam, d_rnorm2, d_vnorm2);
  KernelSquaremRVPoints<<<GridSize(p.npt, block), block>>>(X0.X, X1.X, X2.X, p.npt, r_pt, v_pt,
                                                            d_rnorm2, d_vnorm2);
  CUDA_CHECK(cudaGetLastError());
  Scalar rnorm2, vnorm2;
  CUDA_CHECK(cudaMemcpy(&rnorm2, d_rnorm2, sizeof(Scalar), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&vnorm2, d_vnorm2, sizeof(Scalar), cudaMemcpyDeviceToHost));

  *accepted_extrapolation = false;
  Scalar v_norm = std::sqrt(vnorm2);
  if (v_norm >= (Scalar)1e-30) {
    Scalar alpha = -std::sqrt(rnorm2) / v_norm;
    KernelSquaremRetractCameras<<<GridSize(p.ncam, block), block>>>(X0.R, X0.t, r_cam, v_cam,
                                                                     alpha, p.ncam, Xsq.R, Xsq.t);
    KernelSquaremRetractPoints<<<GridSize(p.npt, block), block>>>(X0.X, r_pt, v_pt, alpha, p.npt,
                                                                   Xsq.X);
    CUDA_CHECK(cudaGetLastError());
    Scalar cost_sq = ComputeCost(p, Xsq, d_cost);
    if (std::isfinite(cost_sq) && cost_sq <= cost2) {
      CopyState(out, Xsq, p.ncam, p.npt);
      *accepted_extrapolation = true;
      return cost_sq;
    }
  }
  CopyState(out, X2, p.ncam, p.npt);
  return cost2;
}

namespace {
void PrintUsage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s --dataset <bal.txt> [--iters N] [--accelerated 0|1]\n"
               "  [--factor F] [--lam L] [--eta E] [--restart-scheme function|gradient]\n"
               "  [--factor-scheme fixed|bb] [--factor-tau T] [--accel-scheme nesterov|squarem]\n"
               "  [--loss l2|huber] [--fp64 0|1]\n"
               "\n"
               "  --dataset      path to a BAL problem file (required)\n"
               "  --iters        outer MM iterations (default 200)\n"
               "  --accelerated  1 = Nesterov + restart (default), 0 = plain MM\n"
               "  --factor       majorization factor (default 2.0, per the spec)\n"
               "  --lam          block-solve Levenberg damping (default 1e-6)\n"
               "  --restart-scheme  function (default) or gradient. function: restart\n"
               "                 when cost_new > EMA(cost) (see --eta). gradient:\n"
               "                 restart when grad_f(y)^T(x_new-x_curr) > 0 (O'Donoghue\n"
               "                 & Candes, arXiv:1204.3982 Sec 3.2 -- momentum vs.\n"
               "                 negative gradient obtuse angle; no extra cost eval,\n"
               "                 reuses grad_f(y) already computed by the block solve).\n"
               "                 Measured near-identical to function scheme on this\n"
               "                 repo's real BA problem at 200 it (6.8544e4 vs\n"
               "                 6.8545e4) -- see CONVERGENCE_LITERATURE.md.\n"
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
               "  --factor-scheme  fixed (default) or bb. bb: adaptive majorization\n"
               "                 factor via a Barzilai-Borwein secant estimate from\n"
               "                 consecutive extrapolation points (our own adaptation\n"
               "                 of the BB principle, not one specific paper's exact\n"
               "                 recipe -- see CONVERGENCE_LITERATURE.md). Measured:\n"
               "                 never beats fixed factor=2.0 on this repo's real BA\n"
               "                 problems at any --factor-tau tested; heavy smoothing\n"
               "                 (low tau) only degenerates back toward the fixed\n"
               "                 baseline, never past it. Kept opt-in, not default.\n"
               "  --factor-tau   EMA smoothing for --factor-scheme bb (default 1.0 =\n"
               "                 raw/unsmoothed secant estimate, noisiest; lower = more\n"
               "                 smoothing toward the previous factor).\n"
               "  --accel-scheme  nesterov (default) or squarem. squarem: an entirely\n"
               "                 separate outer-loop acceleration (Varadhan & Roland\n"
               "                 2008 S3 scheme) that treats one plain MM step as a\n"
               "                 fixed-point map and extrapolates from two consecutive\n"
               "                 applications -- --iters means SQUAREM meta-iterations\n"
               "                 here (2 mm_step calls each), not Nesterov iterations\n"
               "                 (1 each); compare at matched mm_step-equivalent count.\n"
               "                 Uncovered a real finding while building this: even two\n"
               "                 *plain* mm_step calls (factor=2, no SQUAREM math) can\n"
               "                 increase cost on this repo's real BA problem -- the\n"
               "                 fixed majorization factor doesn't universally guarantee\n"
               "                 descent on real data, only along the trajectory plain\n"
               "                 MM happens to visit. See CONVERGENCE_LITERATURE.md for\n"
               "                 the full results and what this implies for the lack of\n"
               "                 adaptive damping in this file's block solve generally.\n"
               "  --damping-scheme  fixed (default) or multilambda. multilambda: per-\n"
               "                 block adaptive damping inspired by\n"
               "                 github.com/msouiai/multishift-bundle-adjustment -- that\n"
               "                 repo solves the full coupled normal equations with CG,\n"
               "                 exploiting Krylov shift-invariance so one matvec stream\n"
               "                 serves a whole grid of LM lambda at once, picking the\n"
               "                 candidate with lowest TRUE cost (their own finding:\n"
               "                 greedy min-cost beats gain-ratio selection). Doesn't\n"
               "                 transplant literally -- DABA's blocks are independent\n"
               "                 6x6/3x3 dense systems, no large system to amortize a\n"
               "                 Krylov trick over -- so here each block tries a small\n"
               "                 lambda grid directly (cheap Cholesky re-solves, not CG),\n"
               "                 evaluates true local cost per candidate, and accepts/\n"
               "                 rejects PER BLOCK (an ill-conditioned block freezes\n"
               "                 instead of moving by a lambda tuned for the average\n"
               "                 block or corrupting neighbors next iteration). Only\n"
               "                 implemented under --accel-scheme nesterov. See\n"
               "                 CONVERGENCE_LITERATURE.md for measured results.\n"
               "  --ml-n-lambda  candidates per block for multilambda (default 6).\n"
               "  --ml-decades   relative lambda grid span as \"lo,hi\" decades around\n"
               "                 each block's own base lambda (default -2,2).\n"
               "  --ml-lam0      initial per-block lambda before any block has adapted\n"
               "                 (default 1e-6, matches --lam's default).\n"
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
  std::string restart_scheme = "function";
  std::string factor_scheme = "fixed";
  Scalar factor_tau = 1.0;
  std::string accel_scheme = "nesterov";
  std::string damping_scheme = "fixed";
  int ml_n_lambda = 6;
  Scalar ml_decade_lo = -2.0, ml_decade_hi = 2.0;
  Scalar ml_lam0 = 1e-6, ml_lam_floor = 1e-12, ml_lam_ceil = 1e12;
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
    else if (a == "--restart-scheme") restart_scheme = next();
    else if (a == "--factor-scheme") factor_scheme = next();
    else if (a == "--factor-tau") factor_tau = std::atof(next().c_str());
    else if (a == "--accel-scheme") accel_scheme = next();
    else if (a == "--damping-scheme") damping_scheme = next();
    else if (a == "--ml-n-lambda") ml_n_lambda = std::atoi(next().c_str());
    else if (a == "--ml-decades") {
      std::string v = next();
      size_t comma = v.find(',');
      ml_decade_lo = std::atof(v.substr(0, comma).c_str());
      ml_decade_hi = std::atof(v.substr(comma + 1).c_str());
    }
    else if (a == "--ml-lam0") ml_lam0 = std::atof(next().c_str());
    else if (a == "--loss") loss = next();
    else if (a == "--fp64") fp64 = std::atoi(next().c_str()) != 0;
    else if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return EXIT_SUCCESS; }
    else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); PrintUsage(argv[0]); return EXIT_FAILURE; }
  }
  if (path.empty()) { PrintUsage(argv[0]); return EXIT_FAILURE; }
  if (restart_scheme != "function" && restart_scheme != "gradient") {
    std::fprintf(stderr, "--restart-scheme must be 'function' or 'gradient', got '%s'\n",
                 restart_scheme.c_str());
    return EXIT_FAILURE;
  }
  if (factor_scheme != "fixed" && factor_scheme != "bb") {
    std::fprintf(stderr, "--factor-scheme must be 'fixed' or 'bb', got '%s'\n",
                 factor_scheme.c_str());
    return EXIT_FAILURE;
  }
  if (accel_scheme != "nesterov" && accel_scheme != "squarem") {
    std::fprintf(stderr, "--accel-scheme must be 'nesterov' or 'squarem', got '%s'\n",
                 accel_scheme.c_str());
    return EXIT_FAILURE;
  }
  if (damping_scheme != "fixed" && damping_scheme != "multilambda") {
    std::fprintf(stderr, "--damping-scheme must be 'fixed' or 'multilambda', got '%s'\n",
                 damping_scheme.c_str());
    return EXIT_FAILURE;
  }
  if (damping_scheme == "multilambda" && accel_scheme != "nesterov") {
    std::fprintf(stderr, "--damping-scheme multilambda only implemented under "
                          "--accel-scheme nesterov\n");
    return EXIT_FAILURE;
  }
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
  DeviceState y_prev = AllocState(bal.ncam, bal.npt);
  CUDA_CHECK(cudaMemcpy(curr.R, R0.data(), 9 * bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(curr.t, t0.data(), 3 * bal.ncam * sizeof(Scalar), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(curr.X, bal.pts.data(), 3 * bal.npt * sizeof(Scalar), cudaMemcpyHostToDevice));
  CopyState(prev, curr, bal.ncam, bal.npt);

  Scalar *Hc, *gc, *Hp, *gp, *gc_prev, *gp_prev, *d_cost, *d_dot, *d_ss, *d_syd;
  CUDA_CHECK(cudaMalloc(&Hc, (size_t)36 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&gc, (size_t)6 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&Hp, (size_t)9 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&gp, (size_t)3 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&gc_prev, (size_t)6 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&gp_prev, (size_t)3 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_cost, sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_dot, sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_ss, sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_syd, sizeof(Scalar)));

  DeviceState squarem_x1 = AllocState(bal.ncam, bal.npt);
  DeviceState squarem_x2 = AllocState(bal.ncam, bal.npt);
  DeviceState squarem_xsq = AllocState(bal.ncam, bal.npt);
  Scalar *r_cam, *v_cam, *r_pt, *v_pt, *d_rnorm2, *d_vnorm2;
  CUDA_CHECK(cudaMalloc(&r_cam, (size_t)6 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&v_cam, (size_t)6 * bal.ncam * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&r_pt, (size_t)3 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&v_pt, (size_t)3 * bal.npt * sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_rnorm2, sizeof(Scalar)));
  CUDA_CHECK(cudaMalloc(&d_vnorm2, sizeof(Scalar)));

  Scalar *ml_base_lam_cam, *ml_base_lam_pt, *ml_lam_mult, *ml_dC_cand, *ml_dX_cand,
      *ml_lam_used_cam, *ml_lam_used_pt, *ml_cam_cost_cand, *ml_pt_cost_cand,
      *ml_cam_cost_now, *ml_pt_cost_now;
  int* ml_d_num_accepted;
  if (damping_scheme == "multilambda") {
    CUDA_CHECK(cudaMalloc(&ml_base_lam_cam, (size_t)bal.ncam * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_base_lam_pt, (size_t)bal.npt * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_lam_mult, (size_t)ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_dC_cand, (size_t)6 * bal.ncam * ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_dX_cand, (size_t)3 * bal.npt * ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_lam_used_cam, (size_t)bal.ncam * ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_lam_used_pt, (size_t)bal.npt * ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_cam_cost_cand, (size_t)bal.ncam * ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_pt_cost_cand, (size_t)bal.npt * ml_n_lambda * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_cam_cost_now, (size_t)bal.ncam * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_pt_cost_now, (size_t)bal.npt * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&ml_d_num_accepted, sizeof(int)));

    std::vector<Scalar> h_base_lam_cam(bal.ncam, ml_lam0), h_base_lam_pt(bal.npt, ml_lam0);
    CUDA_CHECK(cudaMemcpy(ml_base_lam_cam, h_base_lam_cam.data(), bal.ncam * sizeof(Scalar),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ml_base_lam_pt, h_base_lam_pt.data(), bal.npt * sizeof(Scalar),
                          cudaMemcpyHostToDevice));
    std::vector<Scalar> h_lam_mult(ml_n_lambda);
    for (int k = 0; k < ml_n_lambda; ++k) {
      Scalar frac = ml_n_lambda == 1 ? (Scalar)0
                                      : (Scalar)k / (Scalar)(ml_n_lambda - 1);
      Scalar decade = ml_decade_lo + frac * (ml_decade_hi - ml_decade_lo);
      h_lam_mult[k] = std::pow((Scalar)10, decade);
    }
    CUDA_CHECK(cudaMemcpy(ml_lam_mult, h_lam_mult.data(), ml_n_lambda * sizeof(Scalar),
                          cudaMemcpyHostToDevice));
  }

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
  Scalar cur_factor = factor;  // "factor" from here on is the base/reference value for BB
  bool have_y_prev = false;
  CopyState(y_prev, curr, bal.ncam, bal.npt);
  auto t_start = std::chrono::steady_clock::now();
  if (accel_scheme == "squarem") {
    int mmstep_count = 0;
    for (int meta = 1; meta <= n_iter; ++meta) {
      bool accepted_extrap = false;
      cost_curr = SquaremStep(p, curr, next, Hc, gc, Hp, gp, d_cost, factor, lam, squarem_x1,
                               squarem_x2, squarem_xsq, r_cam, v_cam, r_pt, v_pt, d_rnorm2,
                               d_vnorm2, &accepted_extrap);
      mmstep_count += 2;
      CopyState(curr, next, bal.ncam, bal.npt);
      if (meta % 10 == 0 || meta == n_iter) {
        std::printf("  meta%4d cost=%.5e mmstep_equiv=%d choice=%s\n", meta, cost_curr,
                    mmstep_count, accepted_extrap ? "squarem" : "plain2");
      }
    }
    cudaDeviceSynchronize();
    double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("done: %d meta-iters (%d mm_step-equiv), wall=%.3fs (%.2f ms/mm_step-equiv), "
                "final cost=%.6e\n",
                n_iter, mmstep_count, wall, 1000.0 * wall / mmstep_count, cost_curr);
    return 0;
  }
  if (damping_scheme == "multilambda") {
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

      AccumulateGradHess(p, y, Hc, gc, Hp, gp, d_cost);
      int n_cam_acc = 0, n_pt_acc = 0;
      SolveRetractMultiLambda(p, y, next, Hc, gc, Hp, gp, ml_base_lam_cam, ml_base_lam_pt,
                              ml_lam_mult, ml_n_lambda, factor, ml_lam_floor, ml_lam_ceil,
                              ml_dC_cand, ml_dX_cand, ml_lam_used_cam, ml_lam_used_pt,
                              ml_cam_cost_cand, ml_pt_cost_cand, ml_cam_cost_now,
                              ml_pt_cost_now, ml_d_num_accepted, &n_cam_acc, &n_pt_acc);
      Scalar cost_new = ComputeCost(p, next, d_cost);

      bool trigger = false;
      if (accelerated) {
        if (restart_scheme == "gradient") {
          Scalar dot = GradientRestartDot(next, curr, gc, gp, bal.ncam, bal.npt, d_dot);
          trigger = dot > 0;
        } else {
          trigger = cost_new > cost_ema;
        }
      }

      bool restarted = false;
      if (trigger) {
        AccumulateGradHess(p, curr, Hc, gc, Hp, gp, d_cost);
        SolveRetractMultiLambda(p, curr, next, Hc, gc, Hp, gp, ml_base_lam_cam, ml_base_lam_pt,
                                ml_lam_mult, ml_n_lambda, factor, ml_lam_floor, ml_lam_ceil,
                                ml_dC_cand, ml_dX_cand, ml_lam_used_cam, ml_lam_used_pt,
                                ml_cam_cost_cand, ml_pt_cost_cand, ml_cam_cost_now,
                                ml_pt_cost_now, ml_d_num_accepted, &n_cam_acc, &n_pt_acc);
        cost_new = ComputeCost(p, next, d_cost);
        q = std::max(q / (Scalar)2, (Scalar)1);
        restarted = true;
      } else {
        q += 1;
      }

      CopyState(prev, curr, bal.ncam, bal.npt);
      CopyState(curr, next, bal.ncam, bal.npt);
      cost_curr = cost_new;
      cost_ema = (1 - eta) * cost_ema + eta * cost_curr;

      if (it % 10 == 0 || it == n_iter) {
        std::printf("  it%4d cost=%.5e ema=%.5e q=%.3f cam_acc=%d/%d pt_acc=%d/%d "
                    "restart=%d\n",
                    it, cost_curr, cost_ema, q, n_cam_acc, bal.ncam, n_pt_acc, bal.npt,
                    restarted);
      }
    }
    cudaDeviceSynchronize();
    double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("done: %d iters, wall=%.3fs (%.2f ms/iter), final cost=%.6e\n", n_iter, wall,
                1000.0 * wall / n_iter, cost_curr);
    return 0;
  }
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

    AccumulateGradHess(p, y, Hc, gc, Hp, gp, d_cost);

    if (factor_scheme == "bb" && have_y_prev) {
      Scalar ss, syd;
      BBDots(y, y_prev, gc, gc_prev, gp, gp_prev, bal.ncam, bal.npt, d_ss, d_syd, &ss, &syd);
      cur_factor = BBFactor(ss, syd, factor, cur_factor, factor_tau);
    } else {
      cur_factor = factor;
    }
    CopyState(y_prev, y, bal.ncam, bal.npt);
    CUDA_CHECK(cudaMemcpy(gc_prev, gc, (size_t)6 * bal.ncam * sizeof(Scalar),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(gp_prev, gp, (size_t)3 * bal.npt * sizeof(Scalar),
                          cudaMemcpyDeviceToDevice));
    have_y_prev = true;

    SolveRetract(p, y, next, Hc, gc, Hp, gp, cur_factor, lam);
    Scalar cost_new = ComputeCost(p, next, d_cost);

    bool trigger = false;
    if (accelerated) {
      if (restart_scheme == "gradient") {
        Scalar dot = GradientRestartDot(next, curr, gc, gp, bal.ncam, bal.npt, d_dot);
        trigger = dot > 0;
      } else {
        trigger = cost_new > cost_ema;
      }
    }

    bool restarted = false;
    if (trigger) {
      cost_new = MmStep(p, curr, next, Hc, gc, Hp, gp, d_cost, cur_factor, lam);
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
      std::printf("  it%4d cost=%.5e ema=%.5e q=%.3f factor=%.3f restart=%d\n", it, cost_curr,
                  cost_ema, q, cur_factor, restarted);
    }
  }
  cudaDeviceSynchronize();
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::printf("done: %d iters, wall=%.3fs (%.2f ms/iter), final cost=%.6e\n", n_iter, wall,
              1000.0 * wall / n_iter, cost_curr);
  return 0;
}
