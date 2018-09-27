//#ifndef _FLAMEGPU_FUNCTIONS
//#define _FLAMEGPU_FUNCTIONS
#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_


#include "header.h"

 //Environment Bounds
#define MIN_POSITION 0.0f
#define MAX_POSITION 10.0f

//Interaction radius
#define INTERACTION_RADIUS 0.1f
#define SEPARATION_RADIUS 0.005f

//Global Scalers
#define TIME_SCALE 0.0005f
#define GLOBAL_SCALE 0.15f


//Rule scalers
#define STEER_SCALE 0.65f
#define COLLISION_SCALE 0.75f
#define MATCH_SCALE 1.25f


//inline __device__ float dot(glm::vec3 a, glm::vec3 b)
//{
//	return a.x * b.x + a.y * b.y + a.z * b.z;
//}
//
//inline __device__ float length(glm::vec3 v)
//{
//	return sqrtf(dot(v, v));
//}
//
//
//__FLAME_GPU_FUNC__ glm::vec3 boundPosition(glm::vec3 agent_position) {
//	agent_position.x = (agent_position.x < MIN_POSITION) ? MAX_POSITION : agent_position.x;
//	agent_position.x = (agent_position.x > MAX_POSITION) ? MIN_POSITION : agent_position.x;
//
//	agent_position.y = (agent_position.y < MIN_POSITION) ? MAX_POSITION : agent_position.y;
//	agent_position.y = (agent_position.y > MAX_POSITION) ? MIN_POSITION : agent_position.y;
//
//	agent_position.z = (agent_position.z < MIN_POSITION) ? MAX_POSITION : agent_position.z;
//	agent_position.z = (agent_position.z > MAX_POSITION) ? MIN_POSITION : agent_position.z;
//
//	return agent_position;
//}



/**
 * outputedata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_fibril. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function.
 */
__FLAME_GPU_FUNC__ int outputedata(xmachine_memory_fibril* agent, xmachine_message_location_list* location_messages){
        
    add_location_message(location_messages, agent->id, agent->x, agent->y, agent->z);
       
    
    return 0;
}

/**
 * inputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_fibril. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_fibril* agent,
        xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix){
    
    
    // Position within space
    float agent_x = 0.0;
    float agent_y = 0.0;
    float agent_z = 0.0;
    float agent_fx = 0.0;
    float agent_fy = 0.0;
    float agent_fz = 0.0;
    
    //Template for input message iteration
    xmachine_message_location* current_message = get_first_location_message(location_messages, partition_matrix, agent_x, agent_y, agent_z);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
		agent_x = agent_x + 0.1;
		agent_y = agent_y + 0.1;
		agent_z = agent_z - 0.1;

		agent_fx = agent_x * agent_x;
		agent_fy = -agent_y;
		agent_fz = 0.001
        current_message = get_next_location_message(current_message, location_messages, partition_matrix);
    }
	agent->x = agent_x;
	agent->y = agent_y;
	agent->z = agent_z;

	agent->fx = agent_fx;
	agent->fy = agent_fy;
	agent->fz = agent_fz;

//	add_location_message(location_messages, agent->id, agent->x,
//		agent->y, agent->z);

    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
