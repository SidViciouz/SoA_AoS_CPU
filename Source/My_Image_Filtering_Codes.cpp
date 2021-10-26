#include "Context_SoA_AoS.h"

void convert_to_greyscale_image_SoA_CPU(void) {
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

        BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
            + 0.587f * context.SoA_image_input.G_plane[i]  // G
            + 0.114f * context.SoA_image_input.B_plane[i]);  // B
        context.SoA_image_output.R_plane[i] = intensity;
        context.SoA_image_output.G_plane[i] = intensity;
        context.SoA_image_output.B_plane[i] = intensity;
        context.SoA_image_output.A_plane[i] = context.SoA_image_input.A_plane[i];
    }
}

void convert_to_greyscale_image_AoS_CPU(void) {
    Pixel_Channels* tmp_ptr_input = context.AoS_image_input;
    Pixel_Channels* tmp_ptr_output = context.AoS_image_output;
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
        BYTE intensity = BYTE(0.299f * tmp_ptr_input->R // R
            + 0.587f * tmp_ptr_input->G  // G
            + 0.114f * tmp_ptr_input->B);  // B
        tmp_ptr_output->R = intensity;
        tmp_ptr_output->G = intensity;
        tmp_ptr_output->B = intensity;
        tmp_ptr_output->A = tmp_ptr_input->A;

        tmp_ptr_input++; tmp_ptr_output++;
    }
}


char GX[9] = {
    -1,0,1,
    -2,0,2,
    -1,0,1
};

char GY[9] = {
    1,2,1,
    0,0,0,
    -1,-2,-1
};

void convert_to_sobel_image_SoA_CPU(void) {

    BYTE* intensities;
    intensities = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    
    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
        
        BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
            + 0.587f * context.SoA_image_input.G_plane[i]  // G
            + 0.114f * context.SoA_image_input.B_plane[i]);  // B
        intensities[i] = intensity;
        context.SoA_image_output.A_plane[i] = context.SoA_image_input.A_plane[i];
    }

    for (unsigned int i = 0; i < context.image_width; i++) {
        for (unsigned int j = 0; j < context.image_height; j++) {
            char Gx = 0, Gy = 0;
            BYTE G;
            int count = 0;
            int k = j * context.image_width + i;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    if (i + x >= 0 && i + x < context.image_width  && j + y >= 0 && j + y < context.image_height) {
                        unsigned int index = (j+y) * context.image_width + i + x;
                        Gx += GX[count] * intensities[index];
                        Gy += GY[count] * intensities[index];
                    }
                    count++;
                }
            }
            G = (BYTE)sqrt(pow(Gx, 2) + pow(Gy, 2));
            context.SoA_image_output.R_plane[k] = G;
            context.SoA_image_output.G_plane[k] = G;
            context.SoA_image_output.B_plane[k] = G;
        }
    }
    free(intensities);
}
void convert_to_sobel_image_AoS_CPU(void){

    Pixel_Channels* tmp_ptr_input = context.AoS_image_input;
    Pixel_Channels* tmp_ptr_output = context.AoS_image_output;
    BYTE* intensities;

    intensities = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);

    for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
        BYTE intensity = BYTE(0.299f * tmp_ptr_input->R // R
            + 0.587f * tmp_ptr_input->G  // G
            + 0.114f * tmp_ptr_input->B);  // B
        intensities[i] = intensity;
        tmp_ptr_output->A = tmp_ptr_input->A;

        tmp_ptr_input++; tmp_ptr_output++;
    }

    tmp_ptr_output = context.AoS_image_output;
    for (unsigned int j = 0; j < context.image_height; j++) {
        for (unsigned int i = 0; i < context.image_width; i++) {
            char Gx = 0, Gy = 0;
            BYTE G;
            int count = 0;
            int k = j * context.image_width + i;
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    if (i + x >= 0 && i + x < context.image_width && j + y >= 0 && j + y < context.image_height) {
                        unsigned int index = (j + y) * context.image_width + i + x;
                        Gx += GX[count] * intensities[index];
                        Gy += GY[count] * intensities[index];
                    }
                    count++;
                }
            }
            G = (BYTE)sqrt(pow(Gx, 2) + pow(Gy, 2));
            tmp_ptr_output->R = G;
            tmp_ptr_output->G = G;
            tmp_ptr_output->B = G;
            tmp_ptr_output++;
        }
    }
    free(intensities);
}

int initialize_OpenCL(void) {
    cl_int error_code;

    cl_uint n_platforms;
    cl_platform_id* platform_ids;

    cl_uint n_devices;
    cl_device_id* device_ids;

    cl_context my_context;

    size_t program_length;
    char* program_source;
    cl_program my_program;

    context.image_data_bytes = context.image_width * context.image_height * sizeof(unsigned char) * 4;

    error_code = clGetPlatformIDs(NULL, NULL, &n_platforms);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    platform_ids = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platforms);
    error_code = clGetPlatformIDs(n_platforms, platform_ids, NULL);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    error_code = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, NULL, NULL, &n_devices);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    device_ids = (cl_device_id*)malloc(sizeof(cl_device_id) * n_devices);
    error_code = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU,n_devices, device_ids, NULL);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    my_context = clCreateContext(NULL, 1, device_ids, NULL, NULL, &error_code);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    context.my_queue = clCreateCommandQueue(my_context, device_ids[0],CL_QUEUE_PROFILING_ENABLE, &error_code);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    program_length = read_kernel_from_file(KERNEL1, &program_source);

    my_program = clCreateProgramWithSource(my_context, 1, (const char**)&program_source, &program_length, &error_code);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    const char options[] = "-cl-std=CL1.2";
    error_code = clBuildProgram(my_program, 1, device_ids, options, NULL, NULL);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    context.my_kernel = clCreateKernel(my_program, KERNELNAME1, &error_code);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    context.input_buffer_object = clCreateBuffer(my_context, CL_MEM_READ_ONLY, context.image_data_bytes, NULL, &error_code);
    if (CHECK_ERROR_CODE(error_code)) return 1;
    context.output_buffer_object = clCreateBuffer(my_context, CL_MEM_WRITE_ONLY, context.image_data_bytes, NULL, &error_code);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    clEnqueueWriteBuffer(context.my_queue, context.input_buffer_object, CL_FALSE, 0,
         context.image_data_bytes, context.input.image_data, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    clFinish(context.my_queue);

    return 0;
}

int set_local_work_size_and_kernel_arguments(void) {
    cl_int error_code;
    
    context.global_work_size[0] = context.image_width;
    context.global_work_size[1] = context.image_height;

    context.local_work_size[0] = 32;
    context.local_work_size[1] = 16;

    error_code = clSetKernelArg(context.my_kernel, 0, sizeof(cl_mem), &context.input_buffer_object);
    error_code |= clSetKernelArg(context.my_kernel, 1, sizeof(cl_mem), &context.output_buffer_object);
    error_code |= clSetKernelArg(context.my_kernel, 2, sizeof(int), &context.image_width);
    error_code |= clSetKernelArg(context.my_kernel, 3, sizeof(int), &context.image_height);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    return 0;
}

int run_OpenCL_kernel(void) {
    cl_int error_code;

    error_code = clEnqueueNDRangeKernel(context.my_queue,context.my_kernel,
        2, NULL, context.global_work_size, context.local_work_size, 0, NULL, &context.event_for_timing);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    clWaitForEvents(1, &context.event_for_timing);

    error_code = clEnqueueReadBuffer(context.my_queue,context.output_buffer_object,CL_TRUE,0,
         context.image_data_bytes, context.output.image_data, 0, NULL, &context.event_for_timing);
    if (CHECK_ERROR_CODE(error_code)) return 1;

    return 0;
}