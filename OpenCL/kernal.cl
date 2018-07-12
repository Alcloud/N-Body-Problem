__kernel void nbody_simple(
    __global float4* pos, 
    __global float4* vel,
    __global float4* pos_new, 
    __global float4* vel_new)
{
    const float STEP =  0.00027397260273972603; // 1/365/10
    const float EPS = 0.000001;
    const float G = 0.03765; // scaled G

    const float4 step = (float4)(STEP, STEP, STEP, 0.0f); 

    // id of this thread
    int gti = get_global_id(0);

    // total number of particles
    int n = get_global_size(0);

    float4 p = pos[gti];
    float4 v = vel[gti];
    float4 a = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int j = 0; j<n; j++) 
    {
        if (j == gti)
            continue;
        float4 pj = pos[j];
        float4 diff = pj - p; // the mass mass will be ignored
        float invr_distance = rsqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + EPS); // 1/sqrt
        a += pj.w * diff * invr_distance * invr_distance * invr_distance;
    }

    a *= G;

    // update in global memory
    pos_new[gti] = p+step * v + 0.5f * step * step * a;
    vel_new[gti] = v + step * a;
}