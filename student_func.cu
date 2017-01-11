//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <iostream>
#include <stdio.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.
   Note: ascending order == smallest to largest
   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.
   Implementing Parallel Radix Sort with CUDA
   ==========================================
   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.
   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there
   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.
 */

__global__ void histogram(unsigned int* in, unsigned int* hist, int n,unsigned int nBins, unsigned int mask, unsigned int current_bits)
{
	extern __shared__ unsigned int s_local_hist[];
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	for(int j = threadIdx.x; j < nBins; j += blockDim.x)
		s_local_hist[j] = 0;
	__syncthreads();
	
	if (i < n)
	{
		unsigned int bin = (in[i] >> current_bits) & mask;
		atomicAdd(&s_local_hist[bin], 1);
	}
	__syncthreads();
	
	for (unsigned int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
		atomicAdd(&hist[bin], s_local_hist[bin]);
}

__global__ void scan(unsigned int *in,unsigned int *out, int n)
{   
	extern __shared__ int blkData[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		blkData[threadIdx.x] = in[i];
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		int left_value;
		if (threadIdx.x >= stride)
			left_value = blkData[threadIdx.x - stride];
		__syncthreads();
		
		if (threadIdx.x >= stride)
			blkData[threadIdx.x] += left_value;
		__syncthreads();
	}

	if (i < n)
		if(i == 0)
			out[i] = 0;
		else
			out[i] = blkData[i - 1];
}

const dim3 hist_blockSize(256);
const dim3 scan_blockSize(256);

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int nBits;
	for (int i = 1; i < 32; i++)
		if (1 << i > scan_blockSize.x)
		{
			nBits = i - 1;
			break;
		}
	
	dim3 hist_gridSize((numElems - 1)/(hist_blockSize.x) + 1);
    unsigned int nBins = 1 << nBits;
    unsigned int *d_hist;
	cudaMalloc(&d_hist, nBins * sizeof(unsigned int));
	
    unsigned int* histScan = new unsigned int[nBins];
	unsigned int *d_histScan;
	cudaMalloc(&d_histScan, nBins * sizeof(unsigned int));
    
    unsigned int* src = new unsigned int[numElems];
    unsigned int* src_pos = new unsigned int[numElems];
    
    unsigned int* des = new unsigned int[numElems];
    unsigned int* des_pos = new unsigned int[numElems];
    
    cudaMemcpy(src, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(src_pos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
	unsigned int *d_src;
	cudaMalloc(&d_src, numElems * sizeof(unsigned int));
	
	unsigned int mask = (1 << nBits) - 1;
    for (unsigned int i = 0; i < sizeof(unsigned int)*8; i += nBits)
    {
		cudaMemcpy(d_src, src, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//Histogram
		cudaMemset(d_hist, 0, nBins * sizeof(unsigned int));
        
		histogram<<<hist_gridSize, hist_blockSize, nBins*sizeof(unsigned int)>>>(d_src, d_hist, numElems, nBins, mask, i);
		
		cudaDeviceSynchronize();
		 
		//Exclusive Scan
		scan<<<1, scan_blockSize, scan_blockSize.x*sizeof(unsigned int)>>>(d_hist, d_histScan, nBins);
		cudaMemcpy(histScan, d_histScan, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		
		cudaDeviceSynchronize();
            
		// Scatter
        for (unsigned int j = 0; j < numElems; j++)
        {
            unsigned int bin = (src[j] >> i) & mask;
            des[histScan[bin]] = src[j];
			des_pos[histScan[bin]] = src_pos[j];
            histScan[bin]++;
        }
        
		//Swap
        for (unsigned int j = 0; j < numElems; j++)
        {
            unsigned int temp = src[j];
            src[j] = des[j];
            des[j] = temp;
			
			temp = src_pos[j];
            src_pos[j] = des_pos[j];
            des_pos[j] = temp;
        }
    }
	
	cudaMemcpy(d_outputVals, src, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputPos, src_pos, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    delete[] histScan;
    delete[] src;
    delete[] src_pos;
    delete[] des;
    delete[] des_pos;
}