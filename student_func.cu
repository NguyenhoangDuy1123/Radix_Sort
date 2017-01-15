#include "reference_calc.cpp"
#include "utils.h"

__global__ void histogram(unsigned int* in, unsigned int* hist, int n,unsigned int nBins, unsigned int mask, unsigned int current_bits)
{
	extern __shared__ unsigned int s_local_hist[];
	
	for(int j = threadIdx.x; j < nBins; j += blockDim.x)
		s_local_hist[j] = 0;
	
	__syncthreads();
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		unsigned int bin = (in[i] >> current_bits) & mask;
		atomicAdd(&s_local_hist[bin], 1);
	}
	__syncthreads();
	
	for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x)
	{
		hist[bin * gridDim.x + blockIdx.x] = s_local_hist[bin];
	}
}

__global__ void scanBlks(unsigned int *in, unsigned int *out, unsigned int n, unsigned int *blkSums)
{

	extern __shared__ int blkData[];
	int i1 = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	int i2 = i1 + blockDim.x;
	if (i1 < n)
		blkData[threadIdx.x] = in[i1];
	if (i2 < n)
		blkData[threadIdx.x + blockDim.x] = in[i2];
	__syncthreads();


	for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
	{
		int blkDataIdx = (threadIdx.x + 1) * 2 * stride - 1; 
		if (blkDataIdx < 2 * blockDim.x)
			blkData[blkDataIdx] += blkData[blkDataIdx - stride];
		__syncthreads();
	}

	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		int blkDataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride; 
		if (blkDataIdx < 2 * blockDim.x)
			blkData[blkDataIdx] += blkData[blkDataIdx - stride];
		__syncthreads();
	}


	if (i1 < n)
		out[i1] = blkData[threadIdx.x];
	if (i2 < n)
		out[i2] = blkData[threadIdx.x + blockDim.x];

	if (blkSums != NULL && threadIdx.x == 0)
		blkSums[blockIdx.x] = blkData[2 * blockDim.x - 1];

}

__global__ void addPrevSum(unsigned int* blkSumsScan, unsigned int* blkScans, unsigned int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
	if (i < n)
	{
		blkScans[i] += blkSumsScan[blockIdx.x];
	}
}

void scanAll(unsigned int *d_in, unsigned int *d_out, unsigned int n, unsigned int blkSize, unsigned int blkDataSize)
{
	if (n <= blkDataSize)
	{
		scanBlks<<<1, blkSize, blkDataSize * sizeof(int)>>>(d_in, d_out, n, NULL);
		cudaDeviceSynchronize();
	}
	else
	{
		unsigned int *d_blkSums;
		unsigned int numBlks = (n - 1) / blkDataSize + 1;
		cudaMalloc(&d_blkSums, numBlks * sizeof(unsigned int));
		scanBlks<<<numBlks, blkSize, blkDataSize * sizeof(unsigned int)>>>(d_in, d_out, n, d_blkSums);
		cudaDeviceSynchronize();
		
		scanAll(d_blkSums, d_blkSums, numBlks, blkSize, blkDataSize);
		
		addPrevSum<<<numBlks - 1, blkDataSize>>>(d_blkSums, d_out, n);
		cudaDeviceSynchronize();
	}	
}

__global__ void exclusive_scan(unsigned int *in,unsigned int *out, int n)
{   
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

 	if (i < n)
 	{
		out[i] -= in[i];
	}
}

__global__ void scatter(unsigned int *in,unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n, unsigned int *d_histScan, unsigned int mask, unsigned int current_bits, unsigned int nBins)
{
	extern __shared__ unsigned int min_Idx[];
	
	for(int j = threadIdx.x; j < nBins; j += blockDim.x)
		min_Idx[j] = n;
	__syncthreads();
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		unsigned int bin = (in[i] >> current_bits) & mask;
		atomicMin(&min_Idx[bin], i);
	}
	__syncthreads();
	
	if(i < n)
	{
		unsigned int bin = (in[i] >> current_bits) & mask;
		out[d_histScan[blockIdx.x + bin*gridDim.x] + i - min_Idx[bin]] = in[i];
		out_pos[d_histScan[blockIdx.x + bin*gridDim.x] + i - min_Idx[bin]] = in_pos[i];
	}
}

__global__ void swap(unsigned int *in, unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n)
	{
		in[i] = in[i] ^ out[i];
		out[i] = in[i] ^ out[i];
		in[i] = in[i] ^ out[i];
		
		in_pos[i] = in_pos[i] ^ out_pos[i];
		out_pos[i] = in_pos[i] ^ out_pos[i];
		in_pos[i] = in_pos[i] ^ out_pos[i];
	}
}

__global__ void pre_sort(unsigned int *in, unsigned int *in_pos, unsigned int *out, unsigned int *out_pos, unsigned int n, unsigned int nBins, unsigned int mask, unsigned int current_bits, unsigned int *d_hist)
{
	extern __shared__ unsigned int pre_sort_blk_data[];
	unsigned int* blk_value = pre_sort_blk_data;
	unsigned int* blk_pos = pre_sort_blk_data + blockDim.x;
	unsigned int* blk_hist = pre_sort_blk_data + 2*blockDim.x;
	unsigned int* blk_Scan = pre_sort_blk_data + nBins + 2*blockDim.x;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		blk_value[threadIdx.x] = in[i];
		blk_pos[threadIdx.x] = in_pos[i];
	}
	__syncthreads();
	
	//Hist
	for(int j = threadIdx.x; j < nBins; j += blockDim.x)
	{
		blk_hist[j] = 0;
		blk_Scan[j] = 0;
	}
	__syncthreads();
	
	unsigned int bin = (blk_value[threadIdx.x] >> current_bits) & mask;
	atomicAdd(&blk_hist[bin], 1);
	atomicAdd(&blk_Scan[bin], 1);
	__syncthreads();
	
	//Scan
	for (int stride = 1; stride < nBins; stride *= 2)
	{	
		for (int k = threadIdx.x; k < nBins; k += blockDim.x)
		{
			int inVal;
			if (k >= stride)
				inVal = blk_Scan[k - stride];
			__syncthreads();
			if (k >= stride)
				blk_Scan[k] += inVal;
			__syncthreads();
		}
	}
	__syncthreads();
	
	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
		blk_Scan[i] -= blk_hist[i];
	__syncthreads();
	
	//Scatter
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < blockDim.x; i++)
		{
			unsigned int bin = (blk_value[i] >> current_bits) & mask;
			out[blk_Scan[bin] + blockIdx.x*blockDim.x] = blk_value[i];
			out_pos[blk_Scan[bin] + blockIdx.x*blockDim.x] = blk_pos[i];
			blk_Scan[bin]++;
		}
	}
}

const dim3 hist_blockSize(257);
const dim3 scan_blockSize(512);
const dim3 swap_blockSize(1024);

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int nBits = 8;
	
	unsigned int nBins = 1 << nBits;
		
	dim3 hist_gridSize((numElems - 1)/(hist_blockSize.x) + 1);
	dim3 scan_gridSize((hist_gridSize.x * nBins - 1)/(scan_blockSize.x) + 1);
	dim3 swap_gridSize((numElems - 1)/(swap_blockSize.x) + 1);
	
    unsigned int *d_hist;
	cudaMalloc(&d_hist, hist_gridSize.x * nBins * sizeof(unsigned int));
	
	unsigned int *d_histScan;
	cudaMalloc(&d_histScan, hist_gridSize.x * nBins * sizeof(unsigned int));
	
	unsigned int mask = (1 << nBits) - 1;
	
    for (unsigned int i = 0; i < sizeof(unsigned int)*8; i += nBits)
    {
		//Pre_sort
		pre_sort<<<hist_gridSize, hist_blockSize, (2*hist_blockSize.x + 2*nBins)*sizeof(unsigned int)>>>(d_inputVals, d_inputPos, d_inputVals, d_inputPos, numElems, nBins, mask, i, d_hist);

		//Histogram
		histogram<<<hist_gridSize, hist_blockSize, nBins*sizeof(unsigned int)>>>(d_inputVals, d_hist, numElems, nBins, mask, i);
	
		//Exclusive Scan
		scanAll(d_hist, d_histScan, hist_gridSize.x * nBins, scan_blockSize.x, 2*scan_blockSize.x);
		exclusive_scan<<<scan_gridSize, scan_blockSize>>>(d_hist, d_histScan, hist_gridSize.x * nBins);

		//Scatter
		scatter<<<hist_gridSize, hist_blockSize, nBins*sizeof(unsigned int)>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems, d_histScan, mask, i, nBins);
		
		//Swap
		swap<<<swap_gridSize, swap_blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
    }
	swap<<<swap_gridSize, swap_blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
}