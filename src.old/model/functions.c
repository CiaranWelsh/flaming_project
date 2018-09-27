
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
//Environment Bounds
#define MIN_POSITION -0.5f
#define MAX_POSITION +0.5f

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

/**
 * outputedata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_fibril. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function.
 */
__FLAME_GPU_FUNC__ int outputedata(xmachine_memory_fibril* agent, xmachine_message_location_list* location_messages){

    /*
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    float z = 0;
    float fx = 0;
    float fy = 0;
    float fz = 0;

    add_location_message(location_messages, id, x, y, z, fx, fy, fz);
    */

    return 0;
}

/**
 * inputdata FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_fibril. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.
 */
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_fibril* agent, xmachine_message_location_list* location_messages){

    /*
    //Template for input message iteration
    xmachine_message_location* current_message = get_first_location_message(location_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE

        current_message = get_next_location_message(current_message, location_messages);
    }
    */

    return 0;
}




#endif //_FLAMEGPU_FUNCTIONS
