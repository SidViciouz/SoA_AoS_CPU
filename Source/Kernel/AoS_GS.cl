__kernel void aos_gs_kernel(
const __global uchar* input_data, __global uchar* output_data,
int n_columns,int n_rows)
{
	int column = get_global_id(0);
	int row = get_global_id(1);
	uchar RGBA[4];
	uchar intensity;
	
	int offset = 4*(n_columns*row + column);
	
	for(int i=0; i<4; i++)
		RGBA[i] = *(input_data + offset + i);
	
	intensity = (uchar)(0.299f*RGBA[2] + 0.587f*RGBA[1] + 0.114f*RGBA[0]);
	
	for(int i=0; i<3; i++)
		*(output_data + offset + i) = intensity;

	*(output_data + offset + 3) = RGBA[3];
}