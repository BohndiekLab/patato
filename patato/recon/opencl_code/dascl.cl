__kernel void cldas (__global float* result, __global float* signal, __global const float* detectors,
float length_scale, int nx, int ny, int nz, float x_0, float y_0, float z_0, float dx, float dy, float dz, int n_det,
int n_samples, __local float* local_sum, int frame_number, int frame_offset)
{
    // Position indices
    int z_index = get_global_id(0) / (nx * ny);
    int y_index = (get_global_id(0) - z_index * nx * ny) / nx;
    int x_index = get_global_id(0) - z_index * nx * ny - y_index * nx;

    // Actual positions:
    float z = z_index * dz + z_0;
    float y = y_index * dy + y_0;
    float x = x_index * dx + x_0;

    // Detector number:
    int det_id = get_global_id(1);
    int loc_id = get_local_id(1);
    int n_det_loc = get_local_size(1);
    int n_groups = get_num_groups(1);

    int offset = (sqrt(pow(x - detectors[det_id * 3], 2) + pow(y - detectors[det_id * 3 + 1], 2) +
                pow(z - detectors[det_id * 3 + 2], 2)))/length_scale;
    local_sum[loc_id] = 0.f;

    if (!(offset < 0 || offset >= n_samples))
    {
        local_sum[loc_id] = signal[(frame_number + frame_offset)*n_samples*n_det + det_id*n_samples + offset];
    }
    // Wait for other local computations to complete
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = (n_det / n_groups)/2; i>0; i >>= 1) {
        if(loc_id < i) {
            local_sum[loc_id] += local_sum[loc_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(loc_id == 0) {
        result[get_global_id(0) + nx*ny*nz*(det_id/n_det_loc) + n_groups*nx*ny*nz*frame_number] = local_sum[0];
    }
}
