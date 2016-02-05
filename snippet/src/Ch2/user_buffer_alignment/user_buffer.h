
// User defined structure that's aligned generically based on
// vendor's implementation and architecture
// In this case, the aligned is 2^5 = 32
typedef struct __attribute__((aligned)) UserData {
    int x;
    int y;
    int z;
    int w;
    char c;
} UserData;



