//
//  main_SoA_AoS.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//
#pragma warning(disable: 4996)
#pragma warning(disable: 6386)

#include "Context_SoA_AoS.h"
#include "Config_SoA_AoS.h"

Context context;

__int64 _start, _freq, _end;
float compute_time;

int main(int argc, char* argv[]) {
    char program_name[] = "Sogang CSEG475_5475 SoA_versus_AoS_Your_Student_ID";
    fprintf(stdout, "\n###  %s\n\n", program_name);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n");
    fprintf(stdout, "### INPUT FILE NAME = \t\t%s\n", INPUT_FILE_NAME);
    fprintf(stdout, "### OUTPUT FILE NAME = \t\t%s\n\n", OUTPUT_FILE_NAME);

    fprintf(stdout, "### IMAGE OPERATION = ");
    switch (IMAGE_OPERATION) {
    case SoA_GS_CPU:
        fprintf(stdout, "Structure of Arrays (GRAYSCALE) on CPU\n");
        break;
    case SoA_SO_CPU:
        fprintf(stdout, "Structure of Arrays (SOBEL) on CPU\n");
        break;
    case AoS_GS_CPU:
        fprintf(stdout, "Arrays of Structures (GRAYSCALE) on CPU\n");
        break;
    case AoS_SO_CPU:
        fprintf(stdout, "Arrays of Structures (SOBEL) on CPU\n");
        break;
    case SoA_GS_GPU:
        fprintf(stdout, "Structure of Arrays (GRAYSCALE) on GPU\n");
        break;
    case SoA_SO_GPU:
        fprintf(stdout, "Structure of Arrays (SOBEL) on GPU\n");
        break;
    case AoS_GS_GPU:
        fprintf(stdout, "Arrays of Structures (GRAYSCALE) on GPU\n");
        break;
    case AoS_SO_GPU:
        fprintf(stdout, "Arrays of Structures (SOBEL) on GPU\n");
        break;
    default:
        fprintf(stderr, "*** Error: unknown image operation...\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n\n");

    read_input_image_from_file32(INPUT_FILE_NAME);   
    prepare_output_image();

    if (IMAGE_OPERATION == SoA_GS_CPU) {
        prepare_SoA_input_and_output();
        CHECK_TIME_START(_start, _freq);
        convert_to_greyscale_image_SoA_CPU();  // Write a GPU version of this function.
        CHECK_TIME_END(_start, _end, _freq, compute_time);
        fprintf(stdout, "\n^^^ Conversion time by host clock = %.3fms\n\n", compute_time);
        convert_SoA_output_to_output_image_data();
        write_output_image_to_file32(OUTPUT_FILE_NAME);

        free(context.SoA_image_input.R_plane);
        free(context.SoA_image_input.G_plane);
        free(context.SoA_image_input.B_plane);
        free(context.SoA_image_input.A_plane);
        free(context.SoA_image_output.R_plane);
        free(context.SoA_image_output.G_plane);
        free(context.SoA_image_output.B_plane);
        free(context.SoA_image_output.A_plane);
    }
    else if (IMAGE_OPERATION == AoS_GS_CPU) {
        prepare_AoS_input_and_output();
        CHECK_TIME_START(_start, _freq);
        convert_to_greyscale_image_AoS_CPU();  // Write a GPU version of this function.
        CHECK_TIME_END(_start, _end, _freq, compute_time);
        fprintf(stdout, "\n^^^ Conversion time by host clock = %.3fms\n\n", compute_time);
        convert_AoS_output_to_output_image_data();
        write_output_image_to_file32(OUTPUT_FILE_NAME);

        free(context.AoS_image_input);
        free(context.AoS_image_output);
    }
    else if (IMAGE_OPERATION == SoA_SO_CPU) {
        prepare_SoA_input_and_output();
        CHECK_TIME_START(_start, _freq);
        convert_to_sobel_image_SoA_CPU();  // Write a GPU version of this function.
        CHECK_TIME_END(_start, _end, _freq, compute_time);
        fprintf(stdout, "\n^^^ Conversion time by host clock = %.3fms\n\n", compute_time);
        convert_SoA_output_to_output_image_data();
        write_output_image_to_file32(OUTPUT_FILE_NAME);

        free(context.SoA_image_input.R_plane);
        free(context.SoA_image_input.G_plane);
        free(context.SoA_image_input.B_plane);
        free(context.SoA_image_input.A_plane);
        free(context.SoA_image_output.R_plane);
        free(context.SoA_image_output.G_plane);
        free(context.SoA_image_output.B_plane);
        free(context.SoA_image_output.A_plane);
    }
    else if (IMAGE_OPERATION == AoS_SO_CPU) {
        prepare_AoS_input_and_output();
        CHECK_TIME_START(_start, _freq);
        convert_to_sobel_image_AoS_CPU();
        CHECK_TIME_END(_start, _end, _freq, compute_time);
        fprintf(stdout, "\n^^^ Conversion time by host clock = %.3fms\n\n", compute_time);
        convert_AoS_output_to_output_image_data();
        write_output_image_to_file32(OUTPUT_FILE_NAME);

        free(context.AoS_image_input);
        free(context.AoS_image_output);
    }
    else if (IMAGE_OPERATION == SoA_GS_GPU) {
        //prepare_SoA_input_and_output();

        initialize_OpenCL();
        set_local_work_size_and_kernel_arguments();
        run_OpenCL_kernel();

        //convert_SoA_output_to_output_image_data();

        write_output_image_to_file32(OUTPUT_FILE_NAME);
        /*
        free(context.SoA_image_input.R_plane);
        free(context.SoA_image_input.G_plane);
        free(context.SoA_image_input.B_plane);
        free(context.SoA_image_input.A_plane);
        free(context.SoA_image_output.R_plane);
        free(context.SoA_image_output.G_plane);
        free(context.SoA_image_output.B_plane);
        free(context.SoA_image_output.A_plane);*/
    }
    else {
        fprintf(stdout, "^^^ Nothing has been done!\n");
    }
    return 0;
}

