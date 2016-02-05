__kernel void rounding(__global float4 *round_input,
                       __global float4 *round_output) {

   round_output[0] = rint(*round_input);      
   round_output[1] = round(*round_input);
   round_output[2] = ceil(*round_input);
   round_output[3] = floor(*round_input);
   round_output[4] = trunc(*round_input);   
}
