gcc -std=c99 -I $CUDAROOT/include -L $CUDAROOT/library -DDEBUG SobelFilter.c -o SobelFilter -I$CUDAROOT -lOpenCL
srun -n 1 --slurmd-debug=4 --gres=gpu ./SobelFilter
