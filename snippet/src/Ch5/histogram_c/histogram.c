#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define DATA_SIZE 1024
#define BIN_SIZE 256

int main(int argc, char** argv) {
    unsigned int* data = (unsigned int*) malloc( DATA_SIZE * sizeof(unsigned int));
    unsigned int* bin  = (unsigned int*) malloc( BIN_SIZE * sizeof(unsigned int));
    memset(data, 0x0, DATA_SIZE * sizeof(unsigned int));
    memset(bin, 0x0, BIN_SIZE * sizeof(unsigned int));

    for( int i = 0; i < DATA_SIZE; i++) {
        int indx = rand() % BIN_SIZE;
        data[i] = indx;
    }

    for( int i = 0; i < DATA_SIZE; ++i) {
       bin[data[i]]++; 
    }

    for( int i = 0; i < BIN_SIZE; i ++) {
        if (bin[i] == 0) continue; else printf("bin[%d] = %d\n", i, bin[i]);
    }    
}
