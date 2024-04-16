#pragma OPENCL EXTENSION cl_khr_fp64: enable

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define BUFFSIZE 1024
// hack from internet - it seems working
double atomic_dadd(__global double *valq, double delta) {
    union {
        double f;
        unsigned long i;
    } old;

    union {
        double f;
        unsigned long i;
    } new1;

    do {
        old.f = *valq;
        new1.f = old.f + delta;
    } while (atom_cmpxchg((volatile __global unsigned long *)valq, old.i, new1.i) != old.i);
    return old.f;
}  

// g - global
// l - local
double dot_product_g(const global double *a,
                const global double *b,
                int n)
{
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += (a[i] * b[i]);
    }
    return res;
}

double sum_abs(const global double *a,
                int n)
{
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += fabs(a[i]);
    }
    return res;
}

// computes using only single workgroup
// optimize for bigger arrays
kernel void sumProd(global const double *a,
            global const double *b,
            global double *result,
            int N)
{
    local double loc_res[BUFFSIZE];
    const int m = get_local_size(0);
    int t = get_local_id(0);
    int i = get_global_id(0);
    if (i == 0)
        *result = 0;

    loc_res[t] = a[i] * b[i];
    for (int stride = m / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < stride) {
            loc_res[t] += loc_res[t + stride];
        }
    }
    if (t == 0) {
        atomic_dadd(result, loc_res[0]);
    }
    if (i == 1) {
        const int n = get_global_size(0);
        double delta = dot_product_g(a + n, b + n, N - n);
        atomic_dadd(result, delta);
    }
}

// better version than initial
kernel void sumMag(global const double *a,
            global double *result,
            int N)
{
    local double res_loc[BUFFSIZE];
    const int m = get_local_size(0);
    int t = get_local_id(0);
    int i = get_global_id(0);
    if (i == 0)
        *result = 0;

    res_loc[t] = fabs(a[i]);
    for (int stride = m / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < stride)
            res_loc[t] += fabs(res_loc[t + stride]);
    }
    if (t == 0) {
        atomic_dadd(result, res_loc[0]);
    }
    if (i == 1) {
        const int n = get_global_size(0);
        double delta = sum_abs(a + n, N - n);
        atomic_dadd(result, delta);
    }
}

kernel void copy(global double *dst,
                const global double *src,
                int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        dst[i] = src[i];
    } else {
        for (int j = n - 1; j < N; j++)
            dst[j] = src[j];
    }
}

// b = a + k * b
kernel void multAdd(global double *b,
                const global double *a,
                double k,
                int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        b[i] = a[i] + k * b[i];
    } else {
        for (int j = n - 1; j < N; j++)
            b[j] = a[j] + k * b[j];
    }
}

// c = a * b
kernel void mult(const global double *a,
                const global double *b,
                global double *c,
                int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        c[i] = a[i] * b[i];
    } else {
        for (int j = n - 1; j < N; j++)
            c[j] = a[j] * b[j];
    }
}

kernel void lduMul(global double *res,
                const global double *psiPtr,
                const global double *lowerPtr,
                const global double *upperPtr,
                const global int *lPtr,
                const global int *uPtr,
                int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        double tmp1 = lowerPtr[i] * psiPtr[lPtr[i]];
        atomic_dadd(&res[uPtr[i]], tmp1);
        double tmp2 = upperPtr[i] * psiPtr[uPtr[i]];
        atomic_dadd(&res[lPtr[i]], tmp2);
    } else {
        for (int j = n - 1; j < N; j++) {
            double tmp1 = lowerPtr[j] * psiPtr[lPtr[j]];
            atomic_dadd(&res[uPtr[j]], tmp1);
            double tmp2 = upperPtr[j] * psiPtr[uPtr[j]];
            atomic_dadd(&res[lPtr[j]], tmp2);
        }
    }
}

// b += a * k
kernel void addMult(const global double *a,
                global double *b,
                double k,
                int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        b[i] += a[i] * k;
    } else {
        for (int j = n - 1; j < N; j++) {
            b[j] += a[j] * k;
        }
    }
}

kernel void inverse(global double *a,
                int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        a[i] = 1 / a[i];
    } else {
        for (int j = n - 1; j < N; j++) {
            a[j] = 1 / a[j];
        }
    }
}
