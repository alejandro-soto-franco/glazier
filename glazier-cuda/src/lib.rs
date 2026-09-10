//! The same sweep on the GPU, one thread per site of a checkerboard colour.
//!
//! Two copy attempts are independent when neither target sits in the other's
//! neighbourhood. Colouring sites by `(x mod 2, y mod 2)` puts same-colour
//! targets two apart, which is outside a Moore neighbourhood, so a colour's
//! attempts can all run at once. One Monte Carlo step is the four colours in
//! turn, which is as many attempts as there are sites, matching the serial
//! engine's count.
//!
//! Two differences from the serial engine are structural rather than
//! incidental. The serial engine draws its targets with replacement and this
//! one visits every site once per step, and a cell's volume is read before the
//! step and written with an atomic during it, so several accepted copies on
//! one cell within a colour each price their move against the same volume. The
//! two engines therefore agree in distribution rather than trajectory, and the
//! comparison in `tests/` is statistical.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use glazier_core::cpu::Simulation;
use glazier_core::model::Model;
use std::sync::Arc;

const KERNEL: &str = r#"
extern "C" {

// A counter-based stream: the draw is a hash of where and when it is taken,
// so no state is read or written. The SplitMix64 finaliser is the mixer, which
// passes the usual statistical batteries on counter input and costs three
// multiplies. Storing xoshiro state instead moved 64 bytes per site per colour,
// which was more traffic than the lattice itself.
__device__ __forceinline__ unsigned long long mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long draw(
    unsigned long long seed, int site, int step, int colour, int index) {
    return mix(seed
        ^ (0xD1B54A32D192ED03ULL * (unsigned long long)site)
        ^ (0xA0761D6478BD642FULL * (unsigned long long)step)
        ^ (0xE7037ED1A0B428DBULL * (unsigned long long)(4 * colour + index)));
}

__device__ __forceinline__ float to_f32(unsigned long long bits) {
    return (float)((bits >> 11) * (1.0 / 9007199254740992.0));
}

__device__ __forceinline__ int wrap(int v, int n) {
    int out = v % n;
    return out < 0 ? out + n : out;
}

__device__ __forceinline__ float fold(float d, float span) {
    if (span <= 1.0f) return 0.0f;
    if (d >  0.5f * span) return d - span;
    if (d < -0.5f * span) return d + span;
    return d;
}

// ---------------------------------------------------------------- population

__global__ void cell_circular_sums(
    const unsigned int *__restrict__ labels,
    float *__restrict__ sums,
    int sites, int width, int height, int depth)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    unsigned int label = labels[site];
    if (label == 0) return;

    int plane = width * height;
    int x = site % width;
    int y = (site % plane) / width;
    int z = site / plane;

    float ax = 6.283185307179586f * (float)x / (float)width;
    float ay = 6.283185307179586f * (float)y / (float)height;
    float az = 6.283185307179586f * (float)z / (float)depth;

    atomicAdd(&sums[label * 7 + 0], 1.0f);
    atomicAdd(&sums[label * 7 + 1], __cosf(ax));
    atomicAdd(&sums[label * 7 + 2], __sinf(ax));
    atomicAdd(&sums[label * 7 + 3], __cosf(ay));
    atomicAdd(&sums[label * 7 + 4], __sinf(ay));
    atomicAdd(&sums[label * 7 + 5], __cosf(az));
    atomicAdd(&sums[label * 7 + 6], __sinf(az));
}

__global__ void cell_moments(
    const unsigned int *__restrict__ labels,
    const float *__restrict__ centroid,
    float *__restrict__ moments,
    int sites, int width, int height, int depth)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    unsigned int label = labels[site];
    if (label == 0) return;

    int plane = width * height;
    float dx = fold((float)(site % width) - centroid[label * 3 + 0], (float)width);
    float dy = fold((float)((site % plane) / width) - centroid[label * 3 + 1], (float)height);
    float dz = fold((float)(site / plane) - centroid[label * 3 + 2], (float)depth);

    atomicAdd(&moments[label * 6 + 0], dx * dx);
    atomicAdd(&moments[label * 6 + 1], dy * dy);
    atomicAdd(&moments[label * 6 + 2], dz * dz);
    atomicAdd(&moments[label * 6 + 3], dx * dy);
    atomicAdd(&moments[label * 6 + 4], dx * dz);
    atomicAdd(&moments[label * 6 + 5], dy * dz);
}

// `plan` is seven floats per label: mode, centroid xyz, axis xyz. Mode 1
// divides into `daughter`, mode 2 dies.
__global__ void apply_population(
    unsigned int *__restrict__ labels,
    const float *__restrict__ plan,
    const unsigned int *__restrict__ daughter,
    int sites, int width, int height, int depth)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    unsigned int label = labels[site];
    if (label == 0) return;

    float mode = plan[label * 7 + 0];
    if (mode == 0.0f) return;
    if (mode == 2.0f) { labels[site] = 0; return; }

    int plane = width * height;
    float dx = fold((float)(site % width) - plan[label * 7 + 1], (float)width);
    float dy = fold((float)((site % plane) / width) - plan[label * 7 + 2], (float)height);
    float dz = fold((float)(site / plane) - plan[label * 7 + 3], (float)depth);

    float along = dx * plan[label * 7 + 4] + dy * plan[label * 7 + 5] + dz * plan[label * 7 + 6];
    if (along > 0.0f) labels[site] = daughter[label];
}

__global__ void recount(
    const unsigned int *__restrict__ labels,
    int *__restrict__ volume,
    int *__restrict__ surface,
    const int *__restrict__ offsets,
    int n_offsets,
    int width, int height, int depth)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    int sites = width * height * depth;
    if (site >= sites) return;

    unsigned int label = labels[site];
    atomicAdd(&volume[label], 1);

    int plane = width * height;
    int x = site % width;
    int y = (site % plane) / width;
    int z = site / plane;

    int bonds = 0;
    for (int k = 0; k < n_offsets; ++k) {
        int nx = wrap(x + offsets[3 * k + 0], width);
        int ny = wrap(y + offsets[3 * k + 1], height);
        int nz = wrap(z + offsets[3 * k + 2], depth);
        if (labels[(nz * height + ny) * width + nx] != label) ++bonds;
    }
    if (bonds > 0) atomicAdd(&surface[label], bonds);
}

// The largest eigenvalue of a symmetric three by three, in closed form.
//
// Power iteration is the clearer way to do this on the host, and inside a
// sweep it would run for every copy attempt. The trigonometric solution is
// exact and takes one acos.
__device__ __forceinline__ float largest_eigenvalue(
    float a00, float a11, float a22, float a01, float a02, float a12)
{
    float p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if (p1 <= 1e-20f) {
        return fmaxf(a00, fmaxf(a11, a22));
    }
    float q = (a00 + a11 + a22) / 3.0f;
    float d0 = a00 - q, d1 = a11 - q, d2 = a22 - q;
    float p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0f * p1;
    float p = sqrtf(p2 / 6.0f);
    if (p <= 1e-20f) return q;

    float b00 = d0 / p, b11 = d1 / p, b22 = d2 / p;
    float b01 = a01 / p, b02 = a02 / p, b12 = a12 / p;
    float det = b00 * (b11 * b22 - b12 * b12)
              - b01 * (b01 * b22 - b12 * b02)
              + b02 * (b01 * b12 - b11 * b02);
    float r = fminf(1.0f, fmaxf(-1.0f, det / 2.0f));
    float phi = acosf(r) / 3.0f;
    return q + 2.0f * p * __cosf(phi);
}

// Major axis from the ten running sums, in the frame the anchor sets.
__device__ __forceinline__ float axis_length(const float *m)
{
    float n = m[0];
    if (n < 2.0f) return 0.0f;
    float mx = m[1] / n, my = m[2] / n, mz = m[3] / n;
    float xx = m[4] / n - mx * mx;
    float yy = m[5] / n - my * my;
    float zz = m[6] / n - mz * mz;
    float xy = m[7] / n - mx * my;
    float xz = m[8] / n - mx * mz;
    float yz = m[9] / n - my * mz;
    float lambda = largest_eigenvalue(xx, yy, zz, xy, xz, yz);
    return 4.0f * sqrtf(fmaxf(lambda, 0.0f));
}

// Running sums per cell, in the frame each cell's anchor sets. Rebuilt every
// step, so the f32 accumulation only ever carries one step of rounding.
__global__ void cell_sums(
    const unsigned int *__restrict__ labels,
    const float *__restrict__ anchor,
    float *__restrict__ moments,
    int sites, int width, int height, int depth)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    unsigned int label = labels[site];
    if (label == 0) return;

    int plane = width * height;
    float x = fold((float)(site % width) - anchor[label * 3 + 0], (float)width);
    float y = fold((float)((site % plane) / width) - anchor[label * 3 + 1], (float)height);
    float z = fold((float)(site / plane) - anchor[label * 3 + 2], (float)depth);

    float *m = moments + label * 10;
    atomicAdd(&m[0], 1.0f);
    atomicAdd(&m[1], x);
    atomicAdd(&m[2], y);
    atomicAdd(&m[3], z);
    atomicAdd(&m[4], x * x);
    atomicAdd(&m[5], y * y);
    atomicAdd(&m[6], z * z);
    atomicAdd(&m[7], x * y);
    atomicAdd(&m[8], x * z);
    atomicAdd(&m[9], y * z);
}

// ---------------------------------------------------------------- fields

__global__ void field_diffuse(
    const float *__restrict__ values,
    float *__restrict__ out,
    int width, int height, int depth,
    float d, float decay)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    int sites = width * height * depth;
    if (site >= sites) return;

    int plane = width * height;
    int x = site % width;
    int y = (site % plane) / width;
    int z = site / plane;

    float here = values[site];
    float sum = values[(z * height + y) * width + wrap(x - 1, width)]
              + values[(z * height + y) * width + wrap(x + 1, width)]
              + values[(z * height + wrap(y - 1, height)) * width + x]
              + values[(z * height + wrap(y + 1, height)) * width + x];
    float faces = 4.0f;
    if (depth > 1) {
        sum += values[(wrap(z - 1, depth) * height + y) * width + x]
             + values[(wrap(z + 1, depth) * height + y) * width + x];
        faces = 6.0f;
    }
    out[site] = here + d * (sum - faces * here) - decay * here;
}

__global__ void field_exchange(
    float *__restrict__ values,
    const unsigned int *__restrict__ labels,
    const unsigned char *__restrict__ cell_type,
    const float *__restrict__ secretion,
    const float *__restrict__ uptake,
    int sites, int n_species)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    unsigned int label = labels[site];
    if (label == 0) return;
    unsigned char type = cell_type[label];

    for (int s = 0; s < n_species; ++s) {
        int at = s * sites + site;
        float v = values[at] + secretion[type * n_species + s];
        v -= uptake[type * n_species + s] * v;
        values[at] = v < 0.0f ? 0.0f : v;
    }
}

// Whether the ring positions in `present` are one piece under the adjacency
// table.
//
// The neighbourhood is at most twenty-six positions, so it fits in a mask and
// the search is bit arithmetic: no array, no local memory, and the whole test
// is a handful of registers.
__device__ __forceinline__ bool one_piece(unsigned int present, const unsigned int *adjacency)
{
    if (present == 0u || (present & (present - 1u)) == 0u) return true;

    unsigned int reached = present & (~present + 1u);
    for (int round = 0; round < 26; ++round) {
        unsigned int grown = reached;
        unsigned int rest = reached;
        while (rest) {
            int k = __ffs(rest) - 1;
            grown |= adjacency[k];
            rest &= rest - 1u;
        }
        grown &= present;
        if (grown == reached) break;
        reached = grown;
    }
    return reached == present;
}

// Geometric mean of the activity over the sites of `label` around `site`,
// counting the site itself. Zero as soon as one of them has forgotten, which
// is what makes the memory a front rather than a haze.
__device__ __forceinline__ float activity_mean(
    const float *__restrict__ activity,
    const unsigned int *__restrict__ labels,
    const int *__restrict__ offsets,
    int n_offsets,
    int site, unsigned int label,
    int width, int height, int depth)
{
    float here = activity[site];
    if (here <= 0.0f) return 0.0f;
    float log_sum = __logf(here);
    float count = 1.0f;

    int plane = width * height;
    int x = site % width;
    int y = (site % plane) / width;
    int z = site / plane;

    for (int k = 0; k < n_offsets; ++k) {
        int nx = wrap(x + offsets[3 * k + 0], width);
        int ny = wrap(y + offsets[3 * k + 1], height);
        int nz = wrap(z + offsets[3 * k + 2], depth);
        int n = (nz * height + ny) * width + nx;
        if (n == site || labels[n] != label) continue;
        float value = activity[n];
        if (value <= 0.0f) return 0.0f;
        log_sum += __logf(value);
        count += 1.0f;
    }
    return __expf(log_sum / count);
}

__global__ void activity_decay(float *__restrict__ activity, int sites)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= sites) return;
    float v = activity[site];
    activity[site] = v > 0.0f ? v - 1.0f : 0.0f;
}

// ---------------------------------------------------------------- the sweep

__global__ void cpm_sweep(
    unsigned int *__restrict__ labels,
    int *__restrict__ volume,
    const unsigned char *__restrict__ cell_type,
    const float *__restrict__ contact,
    const float *__restrict__ target_volume,
    const float *__restrict__ lambda_volume,
    int *__restrict__ surface,
    const float *__restrict__ target_surface,
    const float *__restrict__ lambda_surface,
    const float *__restrict__ fields,
    const float *__restrict__ chemotaxis,
    float *__restrict__ moments,
    const float *__restrict__ anchor,
    const float *__restrict__ target_length,
    const float *__restrict__ lambda_length,
    int has_length,
    const int *__restrict__ ring,
    const unsigned int *__restrict__ adjacency,
    const unsigned char *__restrict__ connected,
    int ring_size,
    float *__restrict__ activity,
    const float *__restrict__ max_activity,
    const float *__restrict__ lambda_activity,
    const float *__restrict__ external,
    int has_motility,
    int has_external,
    const int *__restrict__ offsets,
    int n_offsets,
    int n_species,
    unsigned long long seed,
    int width, int height, int depth,
    int n_types,
    float temperature,
    int colour, int step)
{
    // Same-colour targets are two apart on every axis, which is outside a
    // twenty-six neighbourhood, so a colour's attempts are independent. A
    // plane needs four colours and a volume eight.
    int half_w = width / 2;
    int half_h = height / 2;
    int half_d = depth > 1 ? depth / 2 : 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_w * half_h * half_d) return;

    int ix = tid % half_w;
    int iy = (tid / half_w) % half_h;
    int iz = tid / (half_w * half_h);

    int x = 2 * ix + (colour & 1);
    int y = 2 * iy + ((colour >> 1) & 1);
    int z = depth > 1 ? 2 * iz + ((colour >> 2) & 1) : 0;
    int target = (z * height + y) * width + x;

    int pick = (int)(draw(seed, target, step, colour, 0) % (unsigned long long)n_offsets);
    int sx = wrap(x + offsets[3 * pick + 0], width);
    int sy = wrap(y + offsets[3 * pick + 1], height);
    int sz = wrap(z + offsets[3 * pick + 2], depth);
    int source = (sz * height + sy) * width + sx;

    unsigned int old_label = labels[target];
    unsigned int new_label = labels[source];
    if (old_label == new_label) return;
    if (old_label != 0 && volume[old_label] <= 1) return;

    unsigned char t_old = cell_type[old_label];
    unsigned char t_new = cell_type[new_label];

    // A copy that would pinch the losing cell in two is refused before it is
    // priced, since no energy term can express a topological veto.
    if (old_label != 0 && ring_size > 0 && connected[t_old]) {
        unsigned int present = 0u;
        for (int k = 0; k < ring_size; ++k) {
            int nx = wrap(x + ring[3 * k + 0], width);
            int ny = wrap(y + ring[3 * k + 1], height);
            int nz = wrap(z + ring[3 * k + 2], depth);
            int n = (nz * height + ny) * width + nx;
            if (n != target && labels[n] == old_label) present |= (1u << k);
        }
        if (!one_piece(present, adjacency)) return;
    }

    float delta = 0.0f;
    int like_old = 0;
    int like_new = 0;
    int bonds = 0;
    for (int k = 0; k < n_offsets; ++k) {
        int nx = wrap(x + offsets[3 * k + 0], width);
        int ny = wrap(y + offsets[3 * k + 1], height);
        int nz = wrap(z + offsets[3 * k + 2], depth);
        int n = (nz * height + ny) * width + nx;
        if (n == target) continue;
        ++bonds;
        unsigned int ln = labels[n];
        unsigned char tn = cell_type[ln];
        if (ln != old_label) delta -= contact[t_old * n_types + tn]; else ++like_old;
        if (ln != new_label) delta += contact[t_new * n_types + tn]; else ++like_new;
    }
    int d_surface_old = 2 * like_old - bonds;
    int d_surface_new = bonds - 2 * like_new;

    if (old_label != 0) {
        float lam = lambda_volume[t_old];
        if (lam != 0.0f) {
            float v = (float)volume[old_label];
            float tv = target_volume[t_old];
            delta += lam * ((v - 1.0f - tv) * (v - 1.0f - tv) - (v - tv) * (v - tv));
        }
    }
    if (new_label != 0) {
        float lam = lambda_volume[t_new];
        if (lam != 0.0f) {
            float v = (float)volume[new_label];
            float tv = target_volume[t_new];
            delta += lam * ((v + 1.0f - tv) * (v + 1.0f - tv) - (v - tv) * (v - tv));
        }
    }

    if (old_label != 0) {
        float lam = lambda_surface[t_old];
        if (lam != 0.0f) {
            float sv = (float)surface[old_label];
            float ts = target_surface[t_old];
            float after = sv + (float)d_surface_old;
            delta += lam * ((after - ts) * (after - ts) - (sv - ts) * (sv - ts));
        }
    }
    if (new_label != 0) {
        float lam = lambda_surface[t_new];
        if (lam != 0.0f) {
            float sv = (float)surface[new_label];
            float ts = target_surface[t_new];
            float after = sv + (float)d_surface_new;
            delta += lam * ((after - ts) * (after - ts) - (sv - ts) * (sv - ts));
        }
    }

    // The length term, read from the running sums the step rebuilt. A site
    // enters each cell's own frame, since a cell straddling the periodic edge
    // has no coordinates in the lattice's.
    if (has_length) {
        float leaving[10];
        float joining[10];
        for (int k = 0; k < 10; ++k) {
            leaving[k] = moments[old_label * 10 + k];
            joining[k] = moments[new_label * 10 + k];
        }
        if (old_label != 0 && lambda_length[t_old] != 0.0f) {
            float ux = fold((float)x - anchor[old_label * 3 + 0], (float)width);
            float uy = fold((float)y - anchor[old_label * 3 + 1], (float)height);
            float uz = fold((float)z - anchor[old_label * 3 + 2], (float)depth);
            float after[10] = {
                leaving[0] - 1.0f, leaving[1] - ux, leaving[2] - uy, leaving[3] - uz,
                leaving[4] - ux * ux, leaving[5] - uy * uy, leaving[6] - uz * uz,
                leaving[7] - ux * uy, leaving[8] - ux * uz, leaving[9] - uy * uz };
            float tl = target_length[t_old];
            float before = axis_length(leaving) - tl;
            float now = axis_length(after) - tl;
            delta += lambda_length[t_old] * (now * now - before * before);
        }
        if (new_label != 0 && lambda_length[t_new] != 0.0f) {
            float ux = fold((float)x - anchor[new_label * 3 + 0], (float)width);
            float uy = fold((float)y - anchor[new_label * 3 + 1], (float)height);
            float uz = fold((float)z - anchor[new_label * 3 + 2], (float)depth);
            float after[10] = {
                joining[0] + 1.0f, joining[1] + ux, joining[2] + uy, joining[3] + uz,
                joining[4] + ux * ux, joining[5] + uy * uy, joining[6] + uz * uz,
                joining[7] + ux * uy, joining[8] + ux * uz, joining[9] + uy * uz };
            float tl = target_length[t_new];
            float before = axis_length(joining) - tl;
            float now = axis_length(after) - tl;
            delta += lambda_length[t_new] * (now * now - before * before);
        }
    }

    // Work against the gradient, read at the two sites the copy runs between.
    // It is a property of the move rather than of the configuration, so it
    // never appears in a total energy.
    if (new_label != 0 && n_species > 0) {
        int sites = width * height * depth;
        for (int s = 0; s < n_species; ++s) {
            float lambda = chemotaxis[t_new * n_species + s];
            if (lambda == 0.0f) continue;
            delta -= lambda * (fields[s * sites + target] - fields[s * sites + source]);
        }
    }

    // Work along the move: the cell's own memory of where it has been, and a
    // constant drift. Neither is a property of the configuration, so neither
    // appears in a total energy.
    if (new_label != 0 && (has_motility || has_external)) {
        if (has_motility && lambda_activity[t_new] != 0.0f && max_activity[t_new] > 0.0f) {
            float into = activity_mean(activity, labels, offsets, n_offsets,
                                       source, new_label, width, height, depth);
            float out_of = activity_mean(activity, labels, offsets, n_offsets,
                                         target, old_label, width, height, depth);
            delta -= lambda_activity[t_new] / max_activity[t_new] * (into - out_of);
        }
        if (has_external) {
            float ex = external[t_new * 3 + 0];
            float ey = external[t_new * 3 + 1];
            float ez = external[t_new * 3 + 2];
            if (ex != 0.0f || ey != 0.0f || ez != 0.0f) {
                delta -= ex * fold((float)(x - sx), (float)width)
                       + ey * fold((float)(y - sy), (float)height)
                       + ez * fold((float)(z - sz), (float)depth);
            }
        }
    }

    bool accept = delta <= 0.0f;
    if (!accept) {
        accept = to_f32(draw(seed, target, step, colour, 1)) < __expf(-delta / temperature);
    }
    if (accept) {
        if (has_motility && new_label != 0 && max_activity[t_new] > 0.0f) {
            activity[target] = max_activity[t_new];
        }
        labels[target] = new_label;
        if (old_label != 0) atomicSub(&volume[old_label], 1);
        if (new_label != 0) atomicAdd(&volume[new_label], 1);
        atomicAdd(&surface[old_label], d_surface_old);
        atomicAdd(&surface[new_label], d_surface_new);
        if (has_length) {
            float ox = fold((float)x - anchor[old_label * 3 + 0], (float)width);
            float oy = fold((float)y - anchor[old_label * 3 + 1], (float)height);
            float oz = fold((float)z - anchor[old_label * 3 + 2], (float)depth);
            float nx2 = fold((float)x - anchor[new_label * 3 + 0], (float)width);
            float ny2 = fold((float)y - anchor[new_label * 3 + 1], (float)height);
            float nz2 = fold((float)z - anchor[new_label * 3 + 2], (float)depth);
            float *mo = moments + old_label * 10;
            float *mn = moments + new_label * 10;
            atomicAdd(&mo[0], -1.0f);      atomicAdd(&mn[0], 1.0f);
            atomicAdd(&mo[1], -ox);        atomicAdd(&mn[1], nx2);
            atomicAdd(&mo[2], -oy);        atomicAdd(&mn[2], ny2);
            atomicAdd(&mo[3], -oz);        atomicAdd(&mn[3], nz2);
            atomicAdd(&mo[4], -ox * ox);   atomicAdd(&mn[4], nx2 * nx2);
            atomicAdd(&mo[5], -oy * oy);   atomicAdd(&mn[5], ny2 * ny2);
            atomicAdd(&mo[6], -oz * oz);   atomicAdd(&mn[6], nz2 * nz2);
            atomicAdd(&mo[7], -ox * oy);   atomicAdd(&mn[7], nx2 * ny2);
            atomicAdd(&mo[8], -ox * oz);   atomicAdd(&mn[8], nx2 * nz2);
            atomicAdd(&mo[9], -oy * oz);   atomicAdd(&mn[9], ny2 * nz2);
        }
    }
}

}
"#;

/// Errors from the GPU backend.
#[derive(Debug)]
pub enum GpuError {
    /// A lattice side is odd, which the checkerboard split cannot cover.
    OddLattice(usize, usize, usize),
    /// The driver, the compiler or a launch reported a failure.
    Driver(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OddLattice(w, h, d) => write!(
                f,
                "the checkerboard split needs even sides; the lattice is {w} by {h} by {d}"
            ),
            Self::Driver(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for GpuError {}

fn driver<E: std::fmt::Debug>(e: E) -> GpuError {
    GpuError::Driver(format!("{e:?}"))
}

/// A run on the device, seeded from a serial run's own state.
pub struct GpuSimulation {
    model: Model,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    diffuse: CudaFunction,
    exchange_kernel: CudaFunction,
    circular_sums: CudaFunction,
    sums_kernel: CudaFunction,
    moments_kernel: CudaFunction,
    apply_population: CudaFunction,
    recount: CudaFunction,
    labels: CudaSlice<u32>,
    volume: CudaSlice<i32>,
    cell_type: CudaSlice<u8>,
    contact: CudaSlice<f32>,
    target_volume: CudaSlice<f32>,
    lambda_volume: CudaSlice<f32>,
    surface: CudaSlice<i32>,
    fields: CudaSlice<f32>,
    scratch: CudaSlice<f32>,
    secretion: CudaSlice<f32>,
    uptake: CudaSlice<f32>,
    chemotaxis: CudaSlice<f32>,
    offsets: CudaSlice<i32>,
    target_length: CudaSlice<f32>,
    lambda_length: CudaSlice<f32>,
    running_moments: CudaSlice<f32>,
    anchor: CudaSlice<f32>,
    ring: CudaSlice<i32>,
    activity: CudaSlice<f32>,
    max_activity: CudaSlice<f32>,
    lambda_activity: CudaSlice<f32>,
    external: CudaSlice<f32>,
    decay_kernel: CudaFunction,
    adjacency: CudaSlice<u32>,
    connected: CudaSlice<u8>,
    ring_size: i32,
    n_offsets: i32,
    colours: i32,
    substeps: Vec<usize>,
    /// Cell types mirrored on the host, since a population decision is per
    /// cell and the array is thousands of entries rather than millions.
    cell_type_host: Vec<u8>,
    rng: glazier_core::rng::Xoshiro,
    target_surface: CudaSlice<f32>,
    lambda_surface: CudaSlice<f32>,
    seed: u64,
    /// Monte Carlo steps completed.
    pub mcs: u64,
}

impl GpuSimulation {
    /// Upload a serial run's state and compile the kernel.
    ///
    /// # Errors
    /// [`GpuError`] if the lattice has an odd side, or if the device rejects
    /// the compile or an allocation.
    pub fn from_cpu(sim: &Simulation) -> Result<Self, GpuError> {
        let (w, h, d) = (sim.model.width, sim.model.height, sim.model.depth);
        if w % 2 != 0 || h % 2 != 0 || (d > 1 && d % 2 != 0) {
            return Err(GpuError::OddLattice(w, h, d));
        }
        // Refusing is the point: a device run that silently dropped the
        // chemistry would still print a tidy summary, and the numbers would be
        // wrong in a way no test on this side would catch.

        let ctx = CudaContext::new(0).map_err(driver)?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNEL).map_err(driver)?;
        let module = ctx.load_module(ptx).map_err(driver)?;
        let kernel = module.load_function("cpm_sweep").map_err(driver)?;
        let diffuse = module.load_function("field_diffuse").map_err(driver)?;
        let exchange_kernel = module.load_function("field_exchange").map_err(driver)?;
        let circular_sums = module.load_function("cell_circular_sums").map_err(driver)?;
        let sums_kernel = module.load_function("cell_sums").map_err(driver)?;
        let decay_kernel = module.load_function("activity_decay").map_err(driver)?;
        let moments_kernel = module.load_function("cell_moments").map_err(driver)?;
        let apply_population = module.load_function("apply_population").map_err(driver)?;
        let recount = module.load_function("recount").map_err(driver)?;

        let n_types = sim.model.n_types();
        let contact: Vec<f32> = sim.model.contact.iter().map(|&v| v as f32).collect();
        let target_volume: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.target_volume as f32)
            .collect();
        let lambda_volume: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.lambda_volume as f32)
            .collect();
        let target_surface: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.target_surface as f32)
            .collect();
        let lambda_surface: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.lambda_surface as f32)
            .collect();
        let target_length: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.target_length as f32)
            .collect();
        let lambda_length: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.lambda_length as f32)
            .collect();
        let labels_count = sim.cell_type.len();

        // The connectivity test reads a fixed neighbourhood and asks only
        // whether the pieces around a site touch, so the adjacency between
        // ring positions is a constant the host works out once.
        let ring_offsets = if sim.model.has_connectivity() {
            glazier_core::connectivity::ring(d)
        } else {
            Vec::new()
        };
        let ring_flat: Vec<i32> = ring_offsets
            .iter()
            .flat_map(|&(ox, oy, oz)| [ox as i32, oy as i32, oz as i32])
            .collect();
        let adjacency: Vec<u32> = ring_offsets
            .iter()
            .map(|&a| {
                ring_offsets
                    .iter()
                    .enumerate()
                    .filter(|&(_, &b)| {
                        (a.0 - b.0).abs() <= 1 && (a.1 - b.1).abs() <= 1 && (a.2 - b.2).abs() <= 1
                    })
                    .fold(0u32, |mask, (index, _)| mask | (1 << index))
            })
            .collect();
        let connected: Vec<u8> = sim
            .model
            .types
            .iter()
            .map(|t| u8::from(t.connected))
            .collect();
        let max_activity: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.max_activity as f32)
            .collect();
        let lambda_activity: Vec<f32> = sim
            .model
            .types
            .iter()
            .map(|t| t.lambda_activity as f32)
            .collect();
        let external: Vec<f32> = sim
            .model
            .types
            .iter()
            .flat_map(|t| t.external.map(|v| v as f32))
            .collect();
        let activity_host: Vec<f32> = sim.activity.values.iter().map(|&v| v as f32).collect();
        let surface_host: Vec<i32> = sim.surface.iter().map(|&v| v as i32).collect();
        debug_assert_eq!(contact.len(), n_types * n_types);

        let volume_host: Vec<i32> = sim.volume.iter().map(|&v| v as i32).collect();

        // Fields run species-major, so one species is a contiguous lattice and
        // a diffusion launch reads a slice rather than a stride.
        let sites = w * h * d;
        let offsets_host: Vec<i32> = sim
            .lattice
            .offsets
            .iter()
            .flat_map(|&(ox, oy, oz)| [ox as i32, oy as i32, oz as i32])
            .collect();
        let n_species = sim.model.species.len();
        let mut fields_host = vec![0.0f32; n_species * sites];
        for (index, values) in sim.fields.values.iter().enumerate() {
            for (site, &v) in values.iter().enumerate() {
                fields_host[index * sites + site] = v as f32;
            }
        }
        let n_types = sim.model.n_types();
        let mut secretion_host = vec![0.0f32; n_types * n_species];
        let mut uptake_host = vec![0.0f32; n_types * n_species];
        let mut chemotaxis_host = vec![0.0f32; n_types * n_species];
        for kind in 0..n_types {
            for s in 0..n_species {
                if let Some(e) = sim.model.exchange.get(kind) {
                    secretion_host[kind * n_species + s] =
                        e.secretion.get(s).copied().unwrap_or(0.0) as f32;
                    uptake_host[kind * n_species + s] =
                        e.uptake.get(s).copied().unwrap_or(0.0) as f32;
                }
                if let Some(row) = sim.model.chemotaxis.get(kind) {
                    chemotaxis_host[kind * n_species + s] =
                        row.get(s).copied().unwrap_or(0.0) as f32;
                }
            }
        }

        Ok(Self {
            model: sim.model.clone(),
            labels: stream.clone_htod(&sim.lattice.labels).map_err(driver)?,
            volume: stream.clone_htod(&volume_host).map_err(driver)?,
            cell_type: stream.clone_htod(&sim.cell_type).map_err(driver)?,
            contact: stream.clone_htod(&contact).map_err(driver)?,
            target_volume: stream.clone_htod(&target_volume).map_err(driver)?,
            lambda_volume: stream.clone_htod(&lambda_volume).map_err(driver)?,
            surface: stream.clone_htod(&surface_host).map_err(driver)?,
            fields: stream.clone_htod(&fields_host).map_err(driver)?,
            scratch: stream.clone_htod(&fields_host).map_err(driver)?,
            secretion: stream.clone_htod(&secretion_host).map_err(driver)?,
            uptake: stream.clone_htod(&uptake_host).map_err(driver)?,
            chemotaxis: stream.clone_htod(&chemotaxis_host).map_err(driver)?,
            n_offsets: sim.lattice.offsets.len() as i32,
            offsets: stream.clone_htod(&offsets_host).map_err(driver)?,
            colours: if d > 1 { 8 } else { 4 },
            substeps: sim.fields.substeps.clone(),
            cell_type_host: sim.cell_type.clone(),
            rng: glazier_core::rng::Xoshiro::seed(sim.model.seed ^ 0x5DEE_CE66_D125_A4B1),
            target_surface: stream.clone_htod(&target_surface).map_err(driver)?,
            lambda_surface: stream.clone_htod(&lambda_surface).map_err(driver)?,
            target_length: stream.clone_htod(&target_length).map_err(driver)?,
            lambda_length: stream.clone_htod(&lambda_length).map_err(driver)?,
            running_moments: stream
                .alloc_zeros::<f32>(labels_count * 10)
                .map_err(driver)?,
            anchor: stream
                .alloc_zeros::<f32>(labels_count * 3)
                .map_err(driver)?,
            ring_size: ring_offsets.len() as i32,
            ring: stream.clone_htod(&ring_flat).map_err(driver)?,
            activity: stream.clone_htod(&activity_host).map_err(driver)?,
            max_activity: stream.clone_htod(&max_activity).map_err(driver)?,
            lambda_activity: stream.clone_htod(&lambda_activity).map_err(driver)?,
            external: stream.clone_htod(&external).map_err(driver)?,
            decay_kernel,
            adjacency: stream.clone_htod(&adjacency).map_err(driver)?,
            connected: stream.clone_htod(&connected).map_err(driver)?,
            seed: sim.model.seed,

            stream,
            kernel,
            diffuse,
            exchange_kernel,
            circular_sums,
            sums_kernel,
            moments_kernel,
            apply_population,
            recount,
            mcs: 0,
        })
    }

    /// Run `steps` Monte Carlo steps, four kernel launches each.
    ///
    /// # Errors
    /// [`GpuError`] if a launch fails.
    pub fn step(&mut self, steps: u64) -> Result<(), GpuError> {
        let (w, h, d) = (self.model.width, self.model.height, self.model.depth);
        let block = 256u32;
        let threads = (w / 2) * (h / 2) * if d > 1 { d / 2 } else { 1 };
        let config = launch_over(threads, block);
        let site_config = launch_over(w * h * d, block);

        let width = w as i32;
        let height = h as i32;
        let depth = d as i32;
        let n_types = self.model.n_types() as i32;
        let n_species = self.model.species.len() as i32;
        let temperature = self.model.temperature as f32;
        let sites = (w * h * d) as i32;

        let has_length = i32::from(self.model.has_length_constraint());
        let has_motility = i32::from(self.model.has_motility());
        let has_external = i32::from(self.model.has_external_potential());
        for _ in 0..steps {
            let step = self.mcs as i32;
            if has_length == 1 {
                self.rebuild_running_moments(block)?;
            }
            for colour in 0..self.colours {
                let mut launch = self.stream.launch_builder(&self.kernel);
                launch
                    .arg(&mut self.labels)
                    .arg(&mut self.volume)
                    .arg(&self.cell_type)
                    .arg(&self.contact)
                    .arg(&self.target_volume)
                    .arg(&self.lambda_volume)
                    .arg(&mut self.surface)
                    .arg(&self.target_surface)
                    .arg(&self.lambda_surface)
                    .arg(&self.fields)
                    .arg(&self.chemotaxis)
                    .arg(&mut self.running_moments)
                    .arg(&self.anchor)
                    .arg(&self.target_length)
                    .arg(&self.lambda_length)
                    .arg(&has_length)
                    .arg(&self.ring)
                    .arg(&self.adjacency)
                    .arg(&self.connected)
                    .arg(&self.ring_size)
                    .arg(&mut self.activity)
                    .arg(&self.max_activity)
                    .arg(&self.lambda_activity)
                    .arg(&self.external)
                    .arg(&has_motility)
                    .arg(&has_external)
                    .arg(&self.offsets)
                    .arg(&self.n_offsets)
                    .arg(&n_species)
                    .arg(&self.seed)
                    .arg(&width)
                    .arg(&height)
                    .arg(&depth)
                    .arg(&n_types)
                    .arg(&temperature)
                    .arg(&colour)
                    .arg(&step);
                unsafe { launch.launch(config) }.map_err(driver)?;
            }
            if has_motility == 1 {
                let mut launch = self.stream.launch_builder(&self.decay_kernel);
                launch.arg(&mut self.activity).arg(&sites);
                unsafe { launch.launch(site_config) }.map_err(driver)?;
            }
            if n_species > 0 {
                self.run_fields(&site_config, width, height, depth, sites, n_species)?;
            }
            if self.model.has_population_events() {
                self.run_population(block)?;
            }
            self.mcs += 1;
        }
        self.stream.synchronize().map_err(driver)?;
        Ok(())
    }

    /// Diffuse, decay and exchange every species by one Monte Carlo step.
    fn run_fields(
        &mut self,
        config: &LaunchConfig,
        width: i32,
        height: i32,
        depth: i32,
        sites: i32,
        n_species: i32,
    ) -> Result<(), GpuError> {
        for index in 0..n_species as usize {
            let steps = self.substeps[index];
            let d = (self.model.species[index].diffusion / steps as f64) as f32;
            let decay = (self.model.species[index].decay / steps as f64) as f32;
            let offset = index * sites as usize;
            for _ in 0..steps {
                let from = self.fields.slice(offset..offset + sites as usize);
                let mut to = self.scratch.slice_mut(offset..offset + sites as usize);
                let mut launch = self.stream.launch_builder(&self.diffuse);
                launch
                    .arg(&from)
                    .arg(&mut to)
                    .arg(&width)
                    .arg(&height)
                    .arg(&depth)
                    .arg(&d)
                    .arg(&decay);
                unsafe { launch.launch(*config) }.map_err(driver)?;
                std::mem::swap(&mut self.fields, &mut self.scratch);
            }
        }

        let mut launch = self.stream.launch_builder(&self.exchange_kernel);
        launch
            .arg(&mut self.fields)
            .arg(&self.labels)
            .arg(&self.cell_type)
            .arg(&self.secretion)
            .arg(&self.uptake)
            .arg(&sites)
            .arg(&n_species);
        unsafe { launch.launch(*config) }.map_err(driver)?;
        Ok(())
    }

    /// Copy the lattice back.
    ///
    /// # Errors
    /// [`GpuError`] if the copy fails.
    pub fn labels(&self) -> Result<Vec<u32>, GpuError> {
        self.stream.clone_dtoh(&self.labels).map_err(driver)
    }

    /// Copy a species' concentrations back.
    ///
    /// # Errors
    /// [`GpuError`] if the copy fails.
    pub fn field(&self, index: usize) -> Result<Vec<f64>, GpuError> {
        let sites = self.model.width * self.model.height * self.model.depth;
        let all = self.stream.clone_dtoh(&self.fields).map_err(driver)?;
        Ok(all[index * sites..(index + 1) * sites]
            .iter()
            .map(|&v| f64::from(v))
            .collect())
    }

    /// Copy the per-cell surfaces back.
    ///
    /// # Errors
    /// [`GpuError`] if the copy fails.
    pub fn surfaces(&self) -> Result<Vec<i64>, GpuError> {
        Ok(self
            .stream
            .clone_dtoh(&self.surface)
            .map_err(driver)?
            .into_iter()
            .map(i64::from)
            .collect())
    }

    /// Copy the per-cell volumes back.
    ///
    /// # Errors
    /// [`GpuError`] if the copy fails.
    pub fn volumes(&self) -> Result<Vec<u32>, GpuError> {
        Ok(self
            .stream
            .clone_dtoh(&self.volume)
            .map_err(driver)?
            .into_iter()
            .map(|v| v.max(0) as u32)
            .collect())
    }
}

impl GpuSimulation {
    /// The anchors and the running moment sums the length term reads.
    ///
    /// Both are rebuilt every step. A cell's anchor is its own centroid, so a
    /// cell straddling the periodic edge still has coordinates to accumulate
    /// against, and rebuilding bounds the f32 accumulation to one step of
    /// rounding rather than a whole run's.
    fn rebuild_running_moments(&mut self, block: u32) -> Result<(), GpuError> {
        let (w, h, d) = (self.model.width, self.model.height, self.model.depth);
        let sites = (w * h * d) as i32;
        let labels = self.cell_type_host.len();
        let extent = [w as f32, h as f32, d as f32];

        let mut sums = self.stream.alloc_zeros::<f32>(labels * 7).map_err(driver)?;
        self.launch_circular_sums(&mut sums, sites, &extent, block)?;
        let sums_host = self.stream.clone_dtoh(&sums).map_err(driver)?;
        let anchor_host = centroids_from(&sums_host, labels, &extent);
        self.anchor = self.stream.clone_htod(&anchor_host).map_err(driver)?;

        self.running_moments = self
            .stream
            .alloc_zeros::<f32>(labels * 10)
            .map_err(driver)?;
        let (width, height, depth) = (w as i32, h as i32, d as i32);
        let mut launch = self.stream.launch_builder(&self.sums_kernel);
        launch
            .arg(&self.labels)
            .arg(&self.anchor)
            .arg(&mut self.running_moments)
            .arg(&sites)
            .arg(&width)
            .arg(&height)
            .arg(&depth);
        unsafe { launch.launch(launch_over(w * h * d, block)) }.map_err(driver)?;
        Ok(())
    }

    /// Division and death for one step.
    ///
    /// The reductions and the relabelling run on the device; the decisions
    /// cross to the host, where one entry per cell is thousands of numbers
    /// rather than millions. A step where nothing divides and nothing dies
    /// still pays for the reductions, so a description with neither skips this
    /// entirely.
    fn run_population(&mut self, block: u32) -> Result<(), GpuError> {
        let (w, h, d) = (self.model.width, self.model.height, self.model.depth);
        let sites = (w * h * d) as i32;
        let labels = self.cell_type_host.len();
        let site_config = launch_over(w * h * d, block);
        let extent = [w as f32, h as f32, d as f32];

        let mut sums = self.stream.alloc_zeros::<f32>(labels * 7).map_err(driver)?;
        self.launch_circular_sums(&mut sums, sites, &extent, block)?;
        let sums_host = self.stream.clone_dtoh(&sums).map_err(driver)?;

        // The circular mean is single-valued whatever the wrap, so its angle
        // gives a centroid for a cell straddling the edge as well.
        let mut centroid_host = vec![0.0f32; labels * 3];
        for label in 1..labels {
            if sums_host[label * 7] <= 0.0 {
                continue;
            }
            for axis in 0..3 {
                let angle =
                    sums_host[label * 7 + 2 + 2 * axis].atan2(sums_host[label * 7 + 1 + 2 * axis]);
                let span = extent[axis];
                centroid_host[label * 3 + axis] =
                    (angle / std::f32::consts::TAU * span).rem_euclid(span);
            }
        }
        let centroid = self.stream.clone_htod(&centroid_host).map_err(driver)?;

        let mut moments = self.stream.alloc_zeros::<f32>(labels * 6).map_err(driver)?;
        self.launch_moments(&centroid, &mut moments, sites, &extent, block)?;
        let moments_host = self.stream.clone_dtoh(&moments).map_err(driver)?;

        let volumes = self.volumes()?;
        let mut plan = vec![0.0f32; labels * 7];
        let mut daughter = vec![0u32; labels];
        let mut new_types: Vec<u8> = Vec::new();
        let mut acted = false;

        for label in 1..labels {
            let volume = f64::from(volumes[label]);
            if volume <= 0.0 {
                continue;
            }
            let spec = self.model.types[self.cell_type_host[label] as usize];
            if spec.division_volume > 0.0 && volume >= spec.division_volume {
                let m: Vec<f64> = (0..6)
                    .map(|i| f64::from(moments_host[label * 6 + i]) / volume)
                    .collect();
                let covariance = [[m[0], m[3], m[4]], [m[3], m[1], m[5]], [m[4], m[5], m[2]]];
                let (axis, _) = glazier_core::moments::principal_axis(covariance);
                plan[label * 7] = 1.0;
                for k in 0..3 {
                    plan[label * 7 + 1 + k] = centroid_host[label * 3 + k];
                    plan[label * 7 + 4 + k] = axis[k] as f32;
                }
                daughter[label] = (labels + new_types.len()) as u32;
                new_types.push(self.cell_type_host[label]);
                acted = true;
            } else if spec.death_rate > 0.0 && self.rng.next_f64() < spec.death_rate {
                plan[label * 7] = 2.0;
                acted = true;
            }
        }

        if !acted {
            return Ok(());
        }

        let width = w as i32;
        let height = h as i32;
        let depth = d as i32;
        let plan_device = self.stream.clone_htod(&plan).map_err(driver)?;
        let daughter_device = self.stream.clone_htod(&daughter).map_err(driver)?;
        let mut launch = self.stream.launch_builder(&self.apply_population);
        launch
            .arg(&mut self.labels)
            .arg(&plan_device)
            .arg(&daughter_device)
            .arg(&sites)
            .arg(&width)
            .arg(&height)
            .arg(&depth);
        unsafe { launch.launch(site_config) }.map_err(driver)?;

        if !new_types.is_empty() {
            self.cell_type_host.extend(new_types);
            self.cell_type = self
                .stream
                .clone_htod(&self.cell_type_host)
                .map_err(driver)?;
            let labels = self.cell_type_host.len();
            self.running_moments = self
                .stream
                .alloc_zeros::<f32>(labels * 10)
                .map_err(driver)?;
            self.anchor = self.stream.alloc_zeros::<f32>(labels * 3).map_err(driver)?;
        }
        self.recount_cells(block)
    }

    fn launch_circular_sums(
        &self,
        sums: &mut CudaSlice<f32>,
        sites: i32,
        extent: &[f32; 3],
        block: u32,
    ) -> Result<(), GpuError> {
        let (width, height, depth) = (extent[0] as i32, extent[1] as i32, extent[2] as i32);
        let mut launch = self.stream.launch_builder(&self.circular_sums);
        launch
            .arg(&self.labels)
            .arg(sums)
            .arg(&sites)
            .arg(&width)
            .arg(&height)
            .arg(&depth);
        unsafe { launch.launch(launch_over(sites as usize, block)) }.map_err(driver)?;
        Ok(())
    }

    fn launch_moments(
        &self,
        centroid: &CudaSlice<f32>,
        moments: &mut CudaSlice<f32>,
        sites: i32,
        extent: &[f32; 3],
        block: u32,
    ) -> Result<(), GpuError> {
        let (width, height, depth) = (extent[0] as i32, extent[1] as i32, extent[2] as i32);
        let mut launch = self.stream.launch_builder(&self.moments_kernel);
        launch
            .arg(&self.labels)
            .arg(centroid)
            .arg(moments)
            .arg(&sites)
            .arg(&width)
            .arg(&height)
            .arg(&depth);
        unsafe { launch.launch(launch_over(sites as usize, block)) }.map_err(driver)?;
        Ok(())
    }

    /// Count every label's volume and surface from the lattice.
    ///
    /// Division and death both change the boundary of every neighbour, so the
    /// counters are rebuilt rather than patched, exactly as the serial engine
    /// rebuilds them.
    fn recount_cells(&mut self, block: u32) -> Result<(), GpuError> {
        let labels = self.cell_type_host.len();
        self.volume = self.stream.alloc_zeros::<i32>(labels).map_err(driver)?;
        self.surface = self.stream.alloc_zeros::<i32>(labels).map_err(driver)?;

        let (w, h, d) = (
            self.model.width as i32,
            self.model.height as i32,
            self.model.depth as i32,
        );
        let mut launch = self.stream.launch_builder(&self.recount);
        launch
            .arg(&self.labels)
            .arg(&mut self.volume)
            .arg(&mut self.surface)
            .arg(&self.offsets)
            .arg(&self.n_offsets)
            .arg(&w)
            .arg(&h)
            .arg(&d);
        unsafe { launch.launch(launch_over((w * h * d) as usize, block)) }.map_err(driver)?;
        Ok(())
    }
}

/// Centroids from the circular sums.
///
/// The mean of `exp(2 pi i x / W)` is single-valued whatever the wrap, so its
/// angle gives a centroid for a cell straddling the edge as well.
fn centroids_from(sums: &[f32], labels: usize, extent: &[f32; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; labels * 3];
    for label in 1..labels {
        if sums[label * 7] <= 0.0 {
            continue;
        }
        for axis in 0..3 {
            let angle = sums[label * 7 + 2 + 2 * axis].atan2(sums[label * 7 + 1 + 2 * axis]);
            let span = extent[axis];
            out[label * 3 + axis] = (angle / std::f32::consts::TAU * span).rem_euclid(span);
        }
    }
    out
}

/// A launch covering `n` items in blocks of `block`.
fn launch_over(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block as usize) as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}
