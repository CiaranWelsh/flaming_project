#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"

//Environment Bounds
#define MIN_POSITION 0f
#define MAX_POSITION 10f

//Interaction radius
#define INTERACTION_RADIUS 1f
#define SEPARATION_RADIUS 0.005f

//Global Scalers
#define TIME_SCALE 0.0005f
#define GLOBAL_SCALE 0.15f


//Rule scalers
#define STEER_SCALE 0.65f
#define COLLISION_SCALE 0.75f
#define MATCH_SCALE 1.25f

inline __device__ float dot(glm::vec3 a, glm::vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length(glm::vec3 v)
{
	return sqrtf(dot(v, v));
}


__FLAME_GPU_FUNC__ glm::vec3 boundPosition(glm::vec3 agent_position) {
	agent_position.x = (agent_position.x < MIN_POSITION) ? MAX_POSITION : agent_position.x;
	agent_position.x = (agent_position.x > MAX_POSITION) ? MIN_POSITION : agent_position.x;

	agent_position.y = (agent_position.y < MIN_POSITION) ? MAX_POSITION : agent_position.y;
	agent_position.y = (agent_position.y > MAX_POSITION) ? MIN_POSITION : agent_position.y;

	agent_position.z = (agent_position.z < MIN_POSITION) ? MAX_POSITION : agent_position.z;
	agent_position.z = (agent_position.z > MAX_POSITION) ? MIN_POSITION : agent_position.z;

	return agent_position;
}

__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_fibril* xmemory, xmachine_message_location_list* location_messages) {
	add_location_message(location_messages, xmemory->id, xmemory->x, xmemory->y, xmemory->z);
	return 0;
}

__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_fibril* xmemory, xmachine_message_location_list* location_messages) {

	//glm::vec3 agent_position = glm::vec3(xmemory->x, xmemory->y, xmemory->z);
	//agent_position++;
	xmemory->x + 0.1;
	xmemory->y + 0.1;
	xmemory->z - 0.1;
	add_location_message(location_messages, xmemory->id, xmemory->x, xmemory->y, xmemory->z);
	return 0;
}

#endif
