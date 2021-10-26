__kernel void soa_gs_kernel(
const __global uchar* input_data1,const __global uchar* input_data2,const __global uchar* input_data3 ,const __global uchar* input_data4
,__global uchar* output_data1,__global uchar* output_data2,__global uchar* output_data3,__global uchar* output_data4,
int n_columns,int n_rows)
{
	int column = get_global_id(0);
	int row = get_global_id(1);
	uchar intensity;
	
	int offset = n_columns*row + column;
	
	intensity = (uchar)(0.299f* *(input_data1 + offset) + 0.587f* *(input_data2 + offset) + 0.114f* *(input_data2 + offset));
	
	*(output_data1 + offset) = intensity;
	*(output_data2 + offset) = intensity;
	*(output_data3 + offset) = intensity;

	*(output_data4 + offset) = *(input_data4 + offset);
}