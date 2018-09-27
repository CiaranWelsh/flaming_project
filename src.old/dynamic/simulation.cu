
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence
 * on www.flamegpu.com website.
 *
 */


  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */

/* fibril Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_fibril_list* d_fibrils;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_fibril_list* d_fibrils_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_fibril_list* d_fibrils_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_fibril_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_fibril_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_fibril_values;  /**< Agent sort identifiers value */

/* fibril state variables */
xmachine_memory_fibril_list* h_fibrils_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_fibril_list* d_fibrils_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_fibril_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_fibrils_default_variable_id_data_iteration;
unsigned int h_fibrils_default_variable_x_data_iteration;
unsigned int h_fibrils_default_variable_y_data_iteration;
unsigned int h_fibrils_default_variable_z_data_iteration;
unsigned int h_fibrils_default_variable_fx_data_iteration;
unsigned int h_fibrils_default_variable_fy_data_iteration;
unsigned int h_fibrils_default_variable_fz_data_iteration;


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_fibril;
size_t temp_scan_storage_bytes_fibril;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Early simulation exit*/
bool g_exit_early;

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** fibril_outputedata
 * Agent function prototype for outputedata function of fibril agent
 */
void fibril_outputedata(cudaStream_t &stream);

/** fibril_inputdata
 * Agent function prototype for inputdata function of fibril agent
 */
void fibril_inputdata(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
		// Initialise some global variables
		g_iterationNumber = 0;
		g_exit_early = false;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_fibrils_default_variable_id_data_iteration = 0;
    h_fibrils_default_variable_x_data_iteration = 0;
    h_fibrils_default_variable_y_data_iteration = 0;
    h_fibrils_default_variable_z_data_iteration = 0;
    h_fibrils_default_variable_fx_data_iteration = 0;
    h_fibrils_default_variable_fy_data_iteration = 0;
    h_fibrils_default_variable_fz_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_fibril_SoA_size = sizeof(xmachine_memory_fibril_list);
	h_fibrils_default = (xmachine_memory_fibril_list*)malloc(xmachine_fibril_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_fibrils_default, &h_xmachine_memory_fibril_default_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* fibril Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_fibrils, xmachine_fibril_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_fibrils_swap, xmachine_fibril_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_fibrils_new, xmachine_fibril_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_fibril_keys, xmachine_memory_fibril_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_fibril_values, xmachine_memory_fibril_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_fibrils_default, xmachine_fibril_SoA_size));
	gpuErrchk( cudaMemcpy( d_fibrils_default, h_fibrils_default, xmachine_fibril_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_fibril = nullptr;
    temp_scan_storage_bytes_fibril = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_fibril, 
        temp_scan_storage_bytes_fibril, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_fibril_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_fibril, temp_scan_storage_bytes_fibril));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

    PROFILE_POP_RANGE();

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_fibril_default_count: %u\n",get_agent_fibril_default_count());
	
#endif
} 


void sort_fibrils_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_fibril_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_fibril_default_count); 
	gridSize = (h_xmachine_memory_fibril_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_fibril_keys, d_xmachine_memory_fibril_values, d_fibrils_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_fibril_keys),  thrust::device_pointer_cast(d_xmachine_memory_fibril_keys) + h_xmachine_memory_fibril_default_count,  thrust::device_pointer_cast(d_xmachine_memory_fibril_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_fibril_agents, no_sm, h_xmachine_memory_fibril_default_count); 
	gridSize = (h_xmachine_memory_fibril_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_fibril_agents<<<gridSize, blockSize>>>(d_xmachine_memory_fibril_values, d_fibrils_default, d_fibrils_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_fibril_list* d_fibrils_temp = d_fibrils_default;
	d_fibrils_default = d_fibrils_swap;
	d_fibrils_swap = d_fibrils_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* fibril Agent variables */
	gpuErrchk(cudaFree(d_fibrils));
	gpuErrchk(cudaFree(d_fibrils_swap));
	gpuErrchk(cudaFree(d_fibrils_new));
	
	free( h_fibrils_default);
	gpuErrchk(cudaFree(d_fibrils_default));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_fibril != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_fibril));
      d_temp_scan_storage_fibril = nullptr;
      temp_scan_storage_bytes_fibril = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

  /* set all non partitioned, spatial partitioned and On-Graph Partitioned message counts to 0*/
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("fibril_inputdata");
	fibril_inputdata(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: fibril_inputdata = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_fibril_default_count: %u\n",get_agent_fibril_default_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* finish whole simulation after this step */
void set_exit_early() {
	g_exit_early = true;
}

bool get_exit_early() {
	return g_exit_early;
}

/* Environment functions */

//host constant declaration



/* Agent data access functions*/

    
int get_agent_fibril_MAX_count(){
    return xmachine_memory_fibril_MAX;
}


int get_agent_fibril_default_count(){
	//continuous agent
	return h_xmachine_memory_fibril_default_count;
	
}

xmachine_memory_fibril_list* get_device_fibril_default_agents(){
	return d_fibrils_default;
}

xmachine_memory_fibril_list* get_host_fibril_default_agents(){
	return h_fibrils_default;
}



/* Host based access of agent variables*/

/** int get_fibril_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_fibril_default_variable_id(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->id,
                    d_fibrils_default->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_fibril_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_fibril_default_variable_x(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->x,
                    d_fibrils_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_fibril_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_fibril_default_variable_y(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->y,
                    d_fibrils_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_fibril_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_fibril_default_variable_z(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->z,
                    d_fibrils_default->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_fibril_default_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_fibril_default_variable_fx(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_fx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->fx,
                    d_fibrils_default->fx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_fx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->fx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fx for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_fibril_default_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_fibril_default_variable_fy(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_fy_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->fy,
                    d_fibrils_default->fy,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_fy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->fy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fy for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_fibril_default_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_fibril_default_variable_fz(unsigned int index){
    unsigned int count = get_agent_fibril_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_fibrils_default_variable_fz_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_fibrils_default->fz,
                    d_fibrils_default->fz,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_fibrils_default_variable_fz_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_fibrils_default->fz[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access fz for the %u th member of fibril_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_fibril_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_fibril_hostToDevice(xmachine_memory_fibril_list * d_dst, xmachine_memory_fibril * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fx, &h_agent->fx, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fy, &h_agent->fy, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fz, &h_agent->fz, sizeof(float), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_fibril_hostToDevice(xmachine_memory_fibril_list * d_dst, xmachine_memory_fibril_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fx, h_src->fx, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fy, h_src->fy, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->fz, h_src->fz, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_fibril* h_allocate_agent_fibril(){
	xmachine_memory_fibril* agent = (xmachine_memory_fibril*)malloc(sizeof(xmachine_memory_fibril));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_fibril));

	return agent;
}
void h_free_agent_fibril(xmachine_memory_fibril** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_fibril** h_allocate_agent_fibril_array(unsigned int count){
	xmachine_memory_fibril ** agents = (xmachine_memory_fibril**)malloc(count * sizeof(xmachine_memory_fibril*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_fibril();
	}
	return agents;
}
void h_free_agent_fibril_array(xmachine_memory_fibril*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_fibril(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_fibril_AoS_to_SoA(xmachine_memory_fibril_list * dst, xmachine_memory_fibril** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->fx[i] = src[i]->fx;
			 
			dst->fy[i] = src[i]->fy;
			 
			dst->fz[i] = src[i]->fz;
			
		}
	}
}


void h_add_agent_fibril_default(xmachine_memory_fibril* agent){
	if (h_xmachine_memory_fibril_count + 1 > xmachine_memory_fibril_MAX){
		printf("Error: Buffer size of fibril agents in state default will be exceeded by h_add_agent_fibril_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_fibril_hostToDevice(d_fibrils_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_fibril_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_fibril_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_fibrils_default, d_fibrils_new, h_xmachine_memory_fibril_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_fibril_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_fibril_default_count, &h_xmachine_memory_fibril_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_fibrils_default_variable_id_data_iteration = 0;
    h_fibrils_default_variable_x_data_iteration = 0;
    h_fibrils_default_variable_y_data_iteration = 0;
    h_fibrils_default_variable_z_data_iteration = 0;
    h_fibrils_default_variable_fx_data_iteration = 0;
    h_fibrils_default_variable_fy_data_iteration = 0;
    h_fibrils_default_variable_fz_data_iteration = 0;
    

}
void h_add_agents_fibril_default(xmachine_memory_fibril** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_fibril_count + count > xmachine_memory_fibril_MAX){
			printf("Error: Buffer size of fibril agents in state default will be exceeded by h_add_agents_fibril_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_fibril_AoS_to_SoA(h_fibrils_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_fibril_hostToDevice(d_fibrils_new, h_fibrils_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_fibril_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_fibril_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_fibrils_default, d_fibrils_new, h_xmachine_memory_fibril_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_fibril_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_fibril_default_count, &h_xmachine_memory_fibril_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_fibrils_default_variable_id_data_iteration = 0;
        h_fibrils_default_variable_x_data_iteration = 0;
        h_fibrils_default_variable_y_data_iteration = 0;
        h_fibrils_default_variable_z_data_iteration = 0;
        h_fibrils_default_variable_fx_data_iteration = 0;
        h_fibrils_default_variable_fy_data_iteration = 0;
        h_fibrils_default_variable_fz_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

int reduce_fibril_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->id),  thrust::device_pointer_cast(d_fibrils_default->id) + h_xmachine_memory_fibril_default_count);
}

int count_fibril_default_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_fibrils_default->id),  thrust::device_pointer_cast(d_fibrils_default->id) + h_xmachine_memory_fibril_default_count, count_value);
}
int min_fibril_default_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_fibril_default_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_fibril_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->x),  thrust::device_pointer_cast(d_fibrils_default->x) + h_xmachine_memory_fibril_default_count);
}

float min_fibril_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_fibril_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_fibril_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->y),  thrust::device_pointer_cast(d_fibrils_default->y) + h_xmachine_memory_fibril_default_count);
}

float min_fibril_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_fibril_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_fibril_default_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->z),  thrust::device_pointer_cast(d_fibrils_default->z) + h_xmachine_memory_fibril_default_count);
}

float min_fibril_default_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_fibril_default_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_fibril_default_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->fx),  thrust::device_pointer_cast(d_fibrils_default->fx) + h_xmachine_memory_fibril_default_count);
}

float min_fibril_default_fx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->fx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_fibril_default_fx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->fx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_fibril_default_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->fy),  thrust::device_pointer_cast(d_fibrils_default->fy) + h_xmachine_memory_fibril_default_count);
}

float min_fibril_default_fy_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->fy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_fibril_default_fy_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->fy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_fibril_default_fz_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_fibrils_default->fz),  thrust::device_pointer_cast(d_fibrils_default->fz) + h_xmachine_memory_fibril_default_count);
}

float min_fibril_default_fz_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->fz);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_fibril_default_fz_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_fibrils_default->fz);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_fibril_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int fibril_outputedata_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** fibril_outputedata
 * Agent function prototype for outputedata function of fibril agent
 */
void fibril_outputedata(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_fibril_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_fibril_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_fibril_list* fibrils_default_temp = d_fibrils;
	d_fibrils = d_fibrils_default;
	d_fibrils_default = fibrils_default_temp;
	//set working count to current state count
	h_xmachine_memory_fibril_count = h_xmachine_memory_fibril_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_fibril_count, &h_xmachine_memory_fibril_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_fibril_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_fibril_default_count, &h_xmachine_memory_fibril_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_fibril_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function outputedata\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_outputedata, fibril_outputedata_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = fibril_outputedata_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (outputedata)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_outputedata<<<g, b, sm_size, stream>>>(d_fibrils, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_fibril_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_fibril_default_count+h_xmachine_memory_fibril_count > xmachine_memory_fibril_MAX){
		printf("Error: Buffer size of outputedata agents in state default will be exceeded moving working agents to next state in function outputedata\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  fibrils_default_temp = d_fibrils;
  d_fibrils = d_fibrils_default;
  d_fibrils_default = fibrils_default_temp;
        
	//update new state agent size
	h_xmachine_memory_fibril_default_count += h_xmachine_memory_fibril_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_fibril_default_count, &h_xmachine_memory_fibril_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int fibril_inputdata_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** fibril_inputdata
 * Agent function prototype for inputdata function of fibril agent
 */
void fibril_inputdata(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_fibril_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_fibril_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_fibril_list* fibrils_default_temp = d_fibrils;
	d_fibrils = d_fibrils_default;
	d_fibrils_default = fibrils_default_temp;
	//set working count to current state count
	h_xmachine_memory_fibril_count = h_xmachine_memory_fibril_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_fibril_count, &h_xmachine_memory_fibril_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_fibril_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_fibril_default_count, &h_xmachine_memory_fibril_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_inputdata, fibril_inputdata_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = fibril_inputdata_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (inputdata)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_inputdata<<<g, b, sm_size, stream>>>(d_fibrils, d_locations);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_fibril_default_count+h_xmachine_memory_fibril_count > xmachine_memory_fibril_MAX){
		printf("Error: Buffer size of inputdata agents in state default will be exceeded moving working agents to next state in function inputdata\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_fibril_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_fibril_Agents<<<gridSize, blockSize, 0, stream>>>(d_fibrils_default, d_fibrils, h_xmachine_memory_fibril_default_count, h_xmachine_memory_fibril_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_fibril_default_count += h_xmachine_memory_fibril_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_fibril_default_count, &h_xmachine_memory_fibril_default_count, sizeof(int)));	
	
	
}


 
extern void reset_fibril_default_count()
{
    h_xmachine_memory_fibril_default_count = 0;
}
