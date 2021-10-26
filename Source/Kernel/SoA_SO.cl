__kernel void soa_so_kernel(
const __global uchar* input_data1,const __global uchar* input_data2,const __global uchar* input_data3,const __global uchar* input_data4,
__global uchar* output_data1,__global uchar* output_data2,__global uchar* output_data3,__global uchar* output_data4,
int n_columns,int n_rows,__constant char* gx_field, __constant char* gy_field
)
{
	int column = get_global_id(0);
	int row = get_global_id(1);
	int count = 0;
	int offset;
	uchar G,intensity;
	float Gx = 0, Gy = 0;
	
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
			if(column + x >= 0 && column + x < n_columns &&
			row + y >= 0 && row + y < n_rows){
				offset = (row+y)*n_columns + column + x;
				intensity = (uchar)(*(input_data1 + offset)*0.299f + *(input_data2 + offset)*0.587f + *(input_data3 + offset)*0.114f);
				Gx += *(gx_field + count)*intensity;
				Gy += *(gy_field + count)*intensity;
			}
			count++;
		}
	}
	
	G = (uchar)sqrt(pow(Gx, 2) + pow(Gy, 2));
	offset = n_columns*row + column;
	*(output_data1 + offset) = G;
	*(output_data2 + offset) = G;
	*(output_data3 + offset) = G;
	*(output_data4 + offset) = *(input_data4 + offset);
}