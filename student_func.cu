//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

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


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int nBits = 2;
    unsigned int nBins = 1 << nBits;
    unsigned int* hist = new unsigned int[nBins];
    unsigned int* histScan = new unsigned int[nBins];
    
    unsigned int* src = new unsigned int[numElems];
    unsigned int* src_pos = new unsigned int[numElems];
    
    unsigned int* des = new unsigned int[numElems];
    unsigned int* des_pos = new unsigned int[numElems];
    
    cudaMemcpy(src, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(src_pos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    for (unsigned int i = 0; i < sizeof(unsigned int)*8; i += nBits)
    {
        unsigned int mask = (1 << nBits) - 1;
        
		//Histogram
        memset(hist, 0, nBins*sizeof(unsigned int));
        
        for (unsigned int j = 0; j < numElems; j++)
        {
            unsigned int bin = (src[j] >> i) & mask;
            hist[bin]++;
        }
        
		//Exclusive Scan
        histScan[0] = 0;
        for (unsigned int j = 1; j < nBins; j++)
            histScan[j] = histScan[j-1] + hist[j-1];
            
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
    
    delete[] hist;
    delete[] histScan;
    delete[] src;
    delete[] src_pos;
    delete[] des;
    delete[] des_pos;
}
