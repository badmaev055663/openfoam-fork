#define BUFFSIZE 1024

// computes using only single workgroup
// optimize for bigger arrays
kernel void sumProd(global const double *a,
            global const double *b,
            global double *result,
            int N)
{
    local double res[BUFFSIZE];
    int t = get_local_id(0);
    if (get_group_id(0))
        return;

    int m = get_local_size(0);
    int sz = N / m;
    res[t] = 0.0;
    for (int j = 0; j < sz; ++j)
    {
        res[t] += (a[j + t * sz] * b[j + t * sz]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    double result_ = 0;
    if (t == 0) {
        int rem = N - m * sz;
        for (int j = 0; j < m; ++j)
            result_ += res[j];
        for (int j = 0; j < rem; ++j)
            result_ += (a[j + m * sz] * b[j + m * sz]);
        *result = result_;
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