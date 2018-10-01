__kernel
void mult(__global const unsigned int *a, __global unsigned int *b, unsigned int size, unsigned int mergesize){
	unsigned int tid = get_global_id(0);
    unsigned int start, m, stop;
    unsigned int i, j, k;
    unsigned int tmp_i, tmp_j;

    start = tid * (mergesize << 1);
    m = start + mergesize;
    if (m >= size) {
        // already sorted, just copy
        for(k = start; k < size; k++) {
            b[k] = a[k];
        }
        return;
    }
    stop = m + mergesize < size ? m + mergesize : size;

    i = start;
    tmp_i = a[i];
    j = m;
    tmp_j = a[j];
    for(k = start; k < stop; k++){
        if(tmp_i < tmp_j) {
            b[k] = tmp_i;
            i++;
            if(i < m) {
                tmp_i = a[i];
            } else {
                // done; copy remaining j
                for(k = k+1; k < stop; k++) {
                    b[k] = a[j];
                    j++;
                }
                return;
            }
        } else {
            b[k] = tmp_j;
            j++;
            if(j < stop) {
                tmp_j = a[j];
            } else {
                // done; copy remaining i
                for(k = k+1; k < stop; k++) {
                    b[k] = a[i];
                    i++;
                }
                return;
            }
        }
    }
}
