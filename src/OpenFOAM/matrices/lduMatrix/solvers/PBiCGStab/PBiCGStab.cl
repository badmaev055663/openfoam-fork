#define BUFFSIZE 1024
kernel void calcSa(global const double *rAPtr,
                global const double *AyAPtr,
                global double *sAPtr,
                double alpha) {
    int i = get_global_id(0);
    sAPtr[i] = rAPtr[i] - alpha * AyAPtr[i];
}