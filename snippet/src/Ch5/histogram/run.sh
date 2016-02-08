gcc -std=c99 -I $CUDAROOT/include -L $CUDAROOT/library -DDEBUG main.c -o Histogram -I$CUDAROOT -lOpenCL
srun -n 1 --slurmd-debug=4 --gres=gpu ./Histogram --height 16 --weight 32