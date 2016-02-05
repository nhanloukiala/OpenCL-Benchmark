#define GROUP_SIZE 256

__kernel 
void bitonicSort(__global uint * data,
                 const uint stage, 
                 const uint subStage,
                 const uint direction) {

    uint threadId = get_global_id(0);
    uint sortIncreasing = direction;
    
    uint distanceBetweenPairs = 1 << (stage - subStage);
    uint blockWidth   = distanceBetweenPairs << 1;

    uint leftId = (threadId % distanceBetweenPairs) + (threadId / distanceBetweenPairs) * blockWidth;

    uint rightId = leftId + distanceBetweenPairs;
    
    uint leftElement = data[leftId];
    uint rightElement = data[rightId];
    
    uint sameDirectionBlockWidth = 1 << stage;
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;

    uint greater;
    uint lesser;
    if(leftElement > rightElement) {
        greater = leftElement;
        lesser  = rightElement;
    } else {
        greater = rightElement;
        lesser  = leftElement;
    }
    
    if(sortIncreasing) {
        data[leftId]  = lesser;
        data[rightId] = greater;
    } else {
        data[leftId]  = greater;
        data[rightId] = lesser;
    }
}

__kernel 
void bitonicSort_sharedmem(__global uint * data,
                           const uint stage, 
                           const uint subStage,
                           const uint direction, 
                           __local uint* sharedMem) {

    uint threadId = get_global_id(0);
    uint sortIncreasing = direction;
    
    uint distanceBetweenPairs = 1 << (stage - subStage);
    uint blockWidth   = distanceBetweenPairs << 1;

    uint leftId = (threadId % distanceBetweenPairs) + (threadId / distanceBetweenPairs) * blockWidth;

    uint rightId = leftId + distanceBetweenPairs;
  
    // Copy data to shared memory on device
    if (threadId == 0) {
        sharedMem[threadId] = data[leftId];
        sharedMem[threadId+1] = data[rightId];
    } else {
        sharedMem[threadId+1] = data[leftId];
        sharedMem[threadId+2] = data[rightId];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sameDirectionBlockWidth = 1 << stage;
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;

    uint greater;
    uint lesser;

    if (threadId == 0) {
        if(sharedMem[threadId] > sharedMem[threadId+1]) {
            greater = sharedMem[threadId];
            lesser  = sharedMem[threadId+1];
        } else {
            greater = sharedMem[threadId+1];
            lesser  = sharedMem[threadId];
        }
    } else {
        if(sharedMem[threadId+1] > sharedMem[threadId+2]) {
            greater = sharedMem[threadId+1];
            lesser  = sharedMem[threadId+2];
        } else {
            greater = sharedMem[threadId+2];
            lesser  = sharedMem[threadId+1];
        }
    } 

    if(sortIncreasing) {
        data[leftId]  = lesser;
        data[rightId] = greater;
    } else {
        data[leftId]  = greater;
        data[rightId] = lesser;
    }
}

__kernel 
void bitonicSort_sharedmem_2(__global uint * data,
                                   const uint stage, 
                                   const uint subStage,
                                   const uint direction, 
                                   __local uint* sharedMem,
                                   __local uint* aux) {

    uint threadId = get_global_id(0);
    uint sortIncreasing = direction;
    
    uint distanceBetweenPairs = 1 << (stage - subStage);
    uint blockWidth   = distanceBetweenPairs << 1;

    uint leftId = (threadId % distanceBetweenPairs) + (threadId / distanceBetweenPairs) * blockWidth;

    uint rightId = leftId + distanceBetweenPairs;
  
    // Copy data to shared memory on device
    if (threadId == 0) {
        sharedMem[threadId] = data[leftId];
        sharedMem[threadId+1] = data[rightId];
    } else {
        sharedMem[threadId+1] = data[leftId];
        sharedMem[threadId+2] = data[rightId];
    }    
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sameDirectionBlockWidth = 1 << stage;
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;

    uint greater;
    uint lesser;

    if (threadId == 0) {
        if(sharedMem[threadId] > sharedMem[threadId+1]) {
            greater = sharedMem[threadId];
            lesser  = sharedMem[threadId+1];
        } else {
            greater = sharedMem[threadId+1];
            lesser  = sharedMem[threadId];
        }
    } else {
        if(sharedMem[threadId+1] > sharedMem[threadId+2]) {
            greater = sharedMem[threadId+1];
            lesser  = sharedMem[threadId+2];
        } else {
            greater = sharedMem[threadId+2];
            lesser  = sharedMem[threadId+1];
        }
    } 

    // Each thread will write the data elements to its own 
    // partition of the shared storage without conflicts.
    const uint stride = 4;
    if(sortIncreasing) {
        //data[leftId]  = lesser;
        //data[rightId] = greater;
        aux[threadId*stride] = leftId; 
        aux[threadId*stride+1] = lesser; 
        aux[threadId*stride+2] = rightId; 
        aux[threadId*stride+3] = greater; 
    } else {
        //data[leftId]  = greater;
        //data[rightId] = lesser;
        aux[threadId*stride] = leftId;
        aux[threadId*stride+1] = greater;
        aux[threadId*stride+2] = rightId;
        aux[threadId*stride+3] = lesser;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(threadId == 0) {
        for(int i = 0; i < GROUP_SIZE * stride; ++i) {
           data[aux[i*stride]] = aux[i*stride+1]; 
           data[aux[i*stride+2]] = aux[i*stride+3]; 
        }
    }
    
}
