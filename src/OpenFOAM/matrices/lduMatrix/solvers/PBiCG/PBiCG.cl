#define BUFFSIZE 2048

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

double sum_l(const local double *a, int n)
{
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += a[i];
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
    local double res_loc[BUFFSIZE];
    int t = get_local_id(0);
    if (get_group_id(0))
        return;

    int m = get_local_size(0);
    int sz = N / m;
    res_loc[t] = dot_product_g(a + t * sz, b + t * sz, sz);

    barrier(CLK_LOCAL_MEM_FENCE);
    double res_glob = 0;
    if (t == 0) {
        int rem = N - m * sz;
        res_glob += sum_l(res_loc, m);
        res_glob += dot_product_g(a + m * sz, b + m * sz, rem);
        *result = res_glob;
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