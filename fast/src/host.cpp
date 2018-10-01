/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
//OpenCL utility layer include
#include "xcl2.hpp"
#include <iostream>
#include <vector>

#define DATA_SIZE 10701
#define COLS 16

using namespace std;
int main(int argc, char** argv)
{
    //Allocate Memory in Host Memory
    size_t vector_size_bytes = sizeof(unsigned int) * DATA_SIZE;

    //Initialize inputs
    std::vector<unsigned int,aligned_allocator<unsigned int>> source_input1     (DATA_SIZE);
    std::vector<unsigned int,aligned_allocator<unsigned int>> source_input2     (DATA_SIZE);
    std::vector<unsigned int,aligned_allocator<unsigned int>> source_hw_results(DATA_SIZE);
    std::vector<unsigned int,aligned_allocator<unsigned int>> source_sw_results(DATA_SIZE);

    // Create the test data and Software Result 
    for(int i = 0 ; i < DATA_SIZE ; i++){
        source_input1[i] = DATA_SIZE - 1 - i;
        source_input2[i] = 0;
        source_hw_results[i] = 0;
        source_sw_results[i] = i;
    }

    //software matrix multiplier

//    for(int i =0; i< DATA_SIZE/COLS ;i++)
//    	 for(int j = 0;j<COLS;j++)
//    	 {   int temp =0;
//    		 for(int k = 0;k< COLS;k++)
//    		 temp = temp + source_input1[i*COLS + k] * source_input2[k*COLS +j];
//    		 source_sw_results[i*COLS +j] = temp;
//
//    	 }

  //  for(int i= 0;i< DATA_SIZE;i++)
   // 	cout<<source_sw_results[i]<<endl;
//OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"mult");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl_mult(program,"mult");

    //Allocate Buffer in Global Memory
    cl::Buffer buffer_input1 (context, CL_MEM_READ_WRITE,
                        vector_size_bytes);
    cl::Buffer buffer_input2 (context, CL_MEM_READ_WRITE,
                           vector_size_bytes);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, 
                            vector_size_bytes);

    //Copy input data to device global memory
    q.enqueueWriteBuffer(buffer_input1, CL_TRUE, 0, vector_size_bytes, source_input1.data());
    q.enqueueWriteBuffer(buffer_input2, CL_TRUE, 0, vector_size_bytes, source_input2.data());

   // int inc = INCR_VALUE;
    int size = DATA_SIZE;
    int cols = COLS;
    //Set the Kernel Arguments
    int narg=0;
    krnl_mult.setArg(2,size);


    cl::Buffer a, b, tmp;
    unsigned int mergesize, num_invocations;

    a = buffer_input1;
    b = buffer_input2;
    for(mergesize = 1; mergesize < size; mergesize <<= 1) {
    	num_invocations = size / (mergesize << 1);
    	if ((mergesize << 1) * num_invocations < size) {
    		num_invocations++;
        }
        //printf("invoking %d times\n\n", num_invocations);

        //printarr(a, size);
        //printarr(b, size);
        krnl_mult.setArg(0,a);
        krnl_mult.setArg(1,b);
        krnl_mult.setArg(3, mergesize);
        q.enqueueNDRangeKernel(krnl_mult,cl::NullRange,cl::NDRange(num_invocations),cl::NullRange);

        //printarr(a, size);
        //printarr(b, size);

        tmp = a;
        a = b;
        b = tmp;

        //printarr(a, size);
        //printarr(b, size);
        }


    //Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(a, CL_TRUE, 0, vector_size_bytes, source_hw_results.data());

    q.finish();

//OPENCL HOST CODE AREA END
    
    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0 ; i < DATA_SIZE ; i++){

    	//printf("%d\n", source_hw_results[i]);

        if (source_hw_results[i] != source_sw_results[i]){
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                << " Device result = " << source_hw_results[i] << std::endl;
            match = false;
          break;
       }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}
