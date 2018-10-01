__kernel
void merge(__global unsigned int* A, __global unsigned int* B, const int mergesize, const int size){
    int i, j, k;
    int start, stop, m;
    int g = get_global_id(0);

    //printf("g=%d, size=%d\n", g, size);

    start = g * (mergesize*2);
    stop = ((start + mergesize * 2 <= size) ? (start + mergesize * 2) : (size))-1;
    m = ((start + mergesize <= size) ? (start + mergesize) : (size))-1;

    for(i=start; i<=m; i++){
        B[i] = A[i];
    }

    for(j=m+1; j<=stop; j++){
        B[m+1+stop-j] = A[j];
    }

    i = start;
    j = stop;

    for(k=start; k<=stop; k++){
        unsigned int tmp_j = B[j];
        unsigned int tmp_i = B[i];
        if(tmp_j < tmp_i) {
            A[k] = tmp_j;
            j--;
        } else {
            A[k] = tmp_i;
            i++;
        }
    }

    //printf("id: %d, mergesize: %d, size: %d, start: %d, stop: %d, m: %d\n", g, mergesize, size, start, stop, m);
}

