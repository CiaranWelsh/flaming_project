
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



#ifndef __HEADER
#define __HEADER

#if defined(__NVCC__) && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__
#define __FLAME_GPU_HOST_FUNC__ __host__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

// FLAME GPU Version Macros.
#define FLAME_GPU_MAJOR_VERSION 1
#define FLAME_GPU_MINOR_VERSION 5
#define FLAME_GPU_PATCH_VERSION 0

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 1024

//Maximum population size of xmachine_memory_fibril
#define xmachine_memory_fibril_MAX 1024


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_location
#define xmachine_message_location_MAX 2048


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_location_partitioningNone

/* Spatial partitioning grid size definitions */

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_fibril
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_fibril
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float z;    /**< X-machine memory variable z of type float.*/
    float fx;    /**< X-machine memory variable fx of type float.*/
    float fy;    /**< X-machine memory variable fy of type float.*/
    float fz;    /**< X-machine memory variable fz of type float.*/
};



/* Message structures */

/** struct xmachine_message_location
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_location
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    float fx;        /**< Message variable fx of type float.*/  
    float fy;        /**< Message variable fy of type float.*/  
    float fz;        /**< Message variable fz of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_fibril_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_fibril_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_fibril_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_fibril_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list y of type float.*/
    float z [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list z of type float.*/
    float fx [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list fx of type float.*/
    float fy [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list fy of type float.*/
    float fz [xmachine_memory_fibril_MAX];    /**< X-machine memory variable list fz of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_location_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_location_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_location_MAX];    /**< Message memory variable list id of type int.*/
    float x [xmachine_message_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_location_MAX];    /**< Message memory variable list z of type float.*/
    float fx [xmachine_message_location_MAX];    /**< Message memory variable list fx of type float.*/
    float fy [xmachine_message_location_MAX];    /**< Message memory variable list fy of type float.*/
    float fz [xmachine_message_location_MAX];    /**< Message memory variable list fz of type float.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * outputedata FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_fibril. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int outputedata(xmachine_memory_fibril* agent, xmachine_message_location_list* location_messages);

/**
 * inputdata FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_fibril. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.
 */
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_fibril* agent, xmachine_message_location_list* location_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) location message implemented in FLAMEGPU_Kernels */

/** add_location_message
 * Function for all types of message partitioning
 * Adds a new location agent to the xmachine_memory_location_list list using a linear mapping
 * @param agents	xmachine_memory_location_list agent list
 * @param id	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param fx	message variable of type float
 * @param fy	message variable of type float
 * @param fz	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_location_message(xmachine_message_location_list* location_messages, int id, float x, float y, float z, float fx, float fy, float fz);
 
/** get_first_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_first_location_message(xmachine_message_location_list* location_messages);

/** get_next_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_next_location_message(xmachine_message_location* current, xmachine_message_location_list* location_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_fibril_agent
 * Adds a new continuous valued fibril agent to the xmachine_memory_fibril_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_fibril_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param z	agent agent variable of type float
 * @param fx	agent agent variable of type float
 * @param fy	agent agent variable of type float
 * @param fz	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_fibril_agent(xmachine_memory_fibril_list* agents, int id, float x, float y, float z, float fx, float fy, float fz);


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_fibrils Pointer to agent list on the host
 * @param d_fibrils Pointer to agent list on the GPU device
 * @param h_xmachine_memory_fibril_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_fibril_list* h_fibrils_default, xmachine_memory_fibril_list* d_fibrils_default, int h_xmachine_memory_fibril_default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_fibrils Pointer to agent list on the host
 * @param h_xmachine_memory_fibril_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_fibril_list* h_fibrils, int* h_xmachine_memory_fibril_count);

/** set_exit_early
 * exits the simulation on the step this is called
 */
extern void set_exit_early();

/** get_exit_early
 * gets whether the simulation is ending this simulation step
 */
extern bool get_exit_early();




/* Return functions used by external code to get agent data from device */

    
/** get_agent_fibril_MAX_count
 * Gets the max agent count for the fibril agent type 
 * @return		the maximum fibril agent count
 */
extern int get_agent_fibril_MAX_count();



/** get_agent_fibril_default_count
 * Gets the agent count for the fibril agent type in state default
 * @return		the current fibril agent count in state default
 */
extern int get_agent_fibril_default_count();

/** reset_default_count
 * Resets the agent count of the fibril in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_fibril_default_count();

/** get_device_fibril_default_agents
 * Gets a pointer to xmachine_memory_fibril_list on the GPU device
 * @return		a xmachine_memory_fibril_list on the GPU device
 */
extern xmachine_memory_fibril_list* get_device_fibril_default_agents();

/** get_host_fibril_default_agents
 * Gets a pointer to xmachine_memory_fibril_list on the CPU host
 * @return		a xmachine_memory_fibril_list on the CPU host
 */
extern xmachine_memory_fibril_list* get_host_fibril_default_agents();


/** sort_fibrils_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_fibrils_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_fibril_list* agents));



/* Host based access of agent variables*/

/** int get_fibril_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_fibril_default_variable_id(unsigned int index);

/** float get_fibril_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_fibril_default_variable_x(unsigned int index);

/** float get_fibril_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_fibril_default_variable_y(unsigned int index);

/** float get_fibril_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_fibril_default_variable_z(unsigned int index);

/** float get_fibril_default_variable_fx(unsigned int index)
 * Gets the value of the fx variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fx
 */
__host__ float get_fibril_default_variable_fx(unsigned int index);

/** float get_fibril_default_variable_fy(unsigned int index)
 * Gets the value of the fy variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fy
 */
__host__ float get_fibril_default_variable_fy(unsigned int index);

/** float get_fibril_default_variable_fz(unsigned int index)
 * Gets the value of the fz variable of an fibril agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable fz
 */
__host__ float get_fibril_default_variable_fz(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_fibril
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated fibril struct.
 */
xmachine_memory_fibril* h_allocate_agent_fibril();
/** h_free_agent_fibril
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_fibril(xmachine_memory_fibril** agent);
/** h_allocate_agent_fibril_array
 * Utility function to allocate an array of structs for  fibril agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_fibril** h_allocate_agent_fibril_array(unsigned int count);
/** h_free_agent_fibril_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_fibril_array(xmachine_memory_fibril*** agents, unsigned int count);


/** h_add_agent_fibril_default
 * Host function to add a single agent of type fibril to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_fibril_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_fibril_default(xmachine_memory_fibril* agent);

/** h_add_agents_fibril_default(
 * Host function to add multiple agents of type fibril to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of fibril agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_fibril_default(xmachine_memory_fibril** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_fibril_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_fibril_default_id_variable();



/** int count_fibril_default_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_fibril_default_id_variable(int count_value);

/** int min_fibril_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_fibril_default_id_variable();
/** int max_fibril_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_fibril_default_id_variable();

/** float reduce_fibril_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_fibril_default_x_variable();



/** float min_fibril_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_fibril_default_x_variable();
/** float max_fibril_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_fibril_default_x_variable();

/** float reduce_fibril_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_fibril_default_y_variable();



/** float min_fibril_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_fibril_default_y_variable();
/** float max_fibril_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_fibril_default_y_variable();

/** float reduce_fibril_default_z_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_fibril_default_z_variable();



/** float min_fibril_default_z_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_fibril_default_z_variable();
/** float max_fibril_default_z_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_fibril_default_z_variable();

/** float reduce_fibril_default_fx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_fibril_default_fx_variable();



/** float min_fibril_default_fx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_fibril_default_fx_variable();
/** float max_fibril_default_fx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_fibril_default_fx_variable();

/** float reduce_fibril_default_fy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_fibril_default_fy_variable();



/** float min_fibril_default_fy_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_fibril_default_fy_variable();
/** float max_fibril_default_fy_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_fibril_default_fy_variable();

/** float reduce_fibril_default_fz_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_fibril_default_fz_variable();



/** float min_fibril_default_fz_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_fibril_default_fz_variable();
/** float max_fibril_default_fz_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_fibril_default_fz_variable();


  
/* global constant variables */


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

