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

kernel void calcSa(global const double *rAPtr,
                global const double *AyAPtr,
                global double *sAPtr,
                double alpha, int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        sAPtr[i] = rAPtr[i] - alpha * AyAPtr[i];
    } else {
        for (int j = n - 1; j < N; j++)
            sAPtr[j] = rAPtr[j] - alpha * AyAPtr[j];
    }
}

kernel void calcPa(global const double *rAPtr,
                global const double *AyAPtr,
                global double *pAPtr,
                double beta, double omega, int N)
{
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        pAPtr[i] = rAPtr[i] + beta * (pAPtr[i] - omega *AyAPtr[i]);
    } else {
        for (int j = n - 1; j < N; j++)
            pAPtr[j] = rAPtr[j] + beta * (pAPtr[j] - omega *AyAPtr[j]);
    }
}