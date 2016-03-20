-std=c99 -DDEBUG work_partition.c -o Partition -framework OpenCL
gcc -std=c99 -Wall -DUNIX -g  -DAPPLE -arch i386 -o Partition MatrixMultiplication.c   -framework OpenCL
./Partition



