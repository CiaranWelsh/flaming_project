
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


#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define RIGHT_BOUND 0.5f
#define LEFT_BOUND -0.5f
#define TOP_BOUND 0.5f
#define BOTTOM_BOUND -0.5f
#define BACK_BOUND -0.5f
#define FRONT_BOUND 0.5f

#define TIME_SCALE 0.01f

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_A. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_A* agent){
//    agent->fx = 1;
    agent->x += TIME_SCALE * agent->fx;
    agent->y += TIME_SCALE * agent->fy;
    agent->z += TIME_SCALE * agent->fz;
    return 0;
}

/**
 * reverse_direction FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_A. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int reverse_direction(xmachine_memory_A* agent){
    if (agent->x > RIGHT_BOUND)
        agent->fx = agent->fx * -1;
    if (agent->x < LEFT_BOUND)
        agent->fx = agent->fx * -1;

    if (agent->y > TOP_BOUND)
        agent->fy = agent->fy * -1;
    if (agent->y < BOTTOM_BOUND)
        agent->fy = agent->fy * -1;

    if (agent->z > FRONT_BOUND)
        agent->fz = agent->fz * -1;
    if (agent->z < BACK_BOUND)
        agent->fz = agent->fz * -1;
//    agent->x = agent->x * -1;
    return 0;
}

/**
 * resume_movement FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_A. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int resume_movement(xmachine_memory_A* agent){
    agent->x += TIME_SCALE * agent->fx;
    agent->y += TIME_SCALE * agent->fy;
    agent->z += TIME_SCALE * agent->fz;
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
