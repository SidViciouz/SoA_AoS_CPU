__kernel void aos_so_kernel(
const __global uchar* input_data, __global uchar* output_data,
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
				offset = 4*(n_columns*(row+y) + column + x);
				intensity = (uchar)(*(input_data + offset)*0.114f + *(input_data + offset + 1)*0.587f + *(input_data + offset + 2)*0.299f);
				Gx += *(gx_field + count)*intensity;
				Gy += *(gy_field + count)*intensity;
			}
			count++;
		}
	}
	
	G = (uchar)sqrt(pow(Gx, 2) + pow(Gy, 2));
	offset = 4*(n_columns*row + column);
	
	for(int i=0; i < 3; i++)
		*(output_data + offset + i) = G;
	*(output_data + offset + 3) = *(input_data + offset + 3);
}