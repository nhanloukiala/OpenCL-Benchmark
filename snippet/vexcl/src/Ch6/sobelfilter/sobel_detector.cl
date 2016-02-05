__kernel void SobelDetector(__global uchar4* input, __global uchar4* output) {
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	float4 Gx = (float4)(0);
	float4 Gy = (float4)(0);

    // Given that we know the (x,y) coordinates of the pixel we're 
    // looking at, its natural to use (x,y) to look at it's neighbouring pixels
    // Convince yourself, that the indexing operation below is doing exactly that

    // the variables i00 through to i22 seek to identify the pixels following
    // the naming convention in graphics programming e.g. OpenGL where i00 refers
    // to the top-left-hand corner and iterates thru to the bottom-right-hand corner

	if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
	{
		float4 i00 = convert_float4(input[(x - 1) + (y - 1) * width]);
		float4 i10 = convert_float4(input[x + (y - 1) * width]);
		float4 i20 = convert_float4(input[(x + 1) + (y - 1) * width]);
		float4 i01 = convert_float4(input[(x - 1) + y * width]);
		float4 i11 = convert_float4(input[x + y * width]);
		float4 i21 = convert_float4(input[(x + 1) + y * width]);
		float4 i02 = convert_float4(input[(x - 1) + (y + 1) * width]);
		float4 i12 = convert_float4(input[x + (y + 1) * width]);
		float4 i22 = convert_float4(input[(x + 1) + (y + 1) * width]);

        // To understand why the masks are applied this way, look at 
        // the mask for Gx and Gy which are respectively equal to the matrices:
        // { {-1, 0, 1}, { {-1,-2,-1},
        //   {-2, 0, 2},   { 0, 0, 0},
        //   {-1, 0, 1}}   { 1, 2, 1}}

		Gx =   i00 + (float4)(2) * i10 + i20 - i02  - (float4)(2) * i12 - i22;

		Gy =   i00 - i20  + (float4)(2)*i01 - (float4)(2)*i21 + i02  -  i22;

        // The math operation here is applied to each element of the unsigned
        // char vector and the final result is applied back to the output image
		output[x + y *width] = convert_uchar4(hypot(Gx, Gy)/(float4)(2));
	}
			
}

	

	 






	

	




	

	

	
	
