#define BUFFSIZE 1024
kernel void calcSa(global const double *rAPtr,
                global const double *AyAPtr,
                global double *sAPtr,
                double alpha, int N) {
    int i = get_global_id(0);
    int n = get_global_size(0);
    if (i < n - 1) {
        sAPtr[i] = rAPtr[i] - alpha * AyAPtr[i];
    } else {
        for (int j = n - 1; j < N; j++)
            sAPtr[j] = rAPtr[j] - alpha * AyAPtr[j];
    }
}