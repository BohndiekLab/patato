__kernel void cldas (__global float* result, __global float* signal, __global const float* detectors,
float length_scale, int nx, int ny, int nz, float x_0, float y_0, float z_0, float dx, float dy, float dz, int n_det,
int n_samples, __local float* local_sum, int frame_number)
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

    int offset = (sqrt(pow(x - detectors[det_id * 3], 2) + pow(y - detectors[det_id * 3 + 1], 2) +
                pow(z - detectors[det_id * 3 + 2], 2)))/length_scale;
    local_sum[loc_id] = 0.f;
    if (!(offset < 0 || offset >= n_samples))
    {
        local_sum[loc_id] = signal[frame_number*n_samples*n_det + det_id*n_samples + offset];
    }
    else{
        local_sum[loc_id] = 0.f;
    }
    // Wait for other local computations to complete
    barrier(CLK_LOCAL_MEM_FENCE);
    // TODO: Implement a more efficient reduce
    if (loc_id == 0){
        float das = 0.f;
        float dmas = 0.f;
        for (int i=0;i<n_det; i++){
            das += local_sum[i];
            for (int j=0; j< n_det; j++){
                if (i == j)
                continue;
                dmas += sqrt(fabs(local_sum[i] * local_sum[j]));
            }
        }
        if (das < 0)
        dmas *= -1;
        result[get_global_id(0) + nx*ny*nz*frame_number] = dmas;
    }
}