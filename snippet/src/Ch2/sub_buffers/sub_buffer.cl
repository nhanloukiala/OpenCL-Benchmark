// OpenCL 1.1 doesn't appear to allow 
// header files to be included in kernel files (*.cl)
typedef struct UserData {
    int x;
    int y;
    int z;
    int w;
} UserData;


__kernel void hello(__global UserData* data) {
    int id = get_global_id(0);
    data[id].w = data[id].x + data[id].y + data[id].z;        
}
