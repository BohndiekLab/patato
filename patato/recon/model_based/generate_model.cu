double sector_integrate(double x, double y, double dx, double r, int qx, int qy){
    // TODO: Add a more detailed description here.
    x = fabs(x);
    y = fabs(y);
    if (y < x){
        double x_temp = x;
        x = y;
        y = x_temp;
    }

    dx = fabs(dx);

    double x_left = x - (1-qx) * dx;
    double x_right = x + qx * dx;
    double y_bottom = y - (1-qy) * dx;
    double y_top = y + qy * dx;

    double angle_1 = 0.;
    double angle_0 = 0.;

    double dangle_1_dr = 0.;
    double dangle_0_dr = 0.;

    double r2 = r * r;
    double x_left2 = x_left * x_left;
    double x_right2 = x_right * x_right;
    double y_bottom2 = y_bottom * y_bottom;
    double y_top2 = y_top * y_top;

    double sa1;
    double ca1;
    double sa0;
    double ca0;
    if (x_left2 + y_bottom2 > r2){
        // outside of square
        return 0.;
    }
    else if (x_right2 + y_top2 < r2){
        // outside of square
        return 0.;
    }
    else if (x_left2 + y_top2 < r2){
        sa1 = y_top/r;
        ca1 = sqrt(1-sa1*sa1);
        // angle_1 = asin(sa1);
        dangle_1_dr = -y_top / (r2 * sqrt(1 - y_top2 / r2));
        ca0 = x_right/r;
        sa0 = sqrt(1-ca0*ca0);
        // angle_0 = acos(ca0);
        dangle_0_dr = x_right / (r2 * sqrt(1 - x_right2 / r2));
    }
    else if (x_right * x_right + y_bottom * y_bottom > r * r){
        ca1 = x_left / r;
        sa1 = sqrt(1-ca1*ca1);
        // angle_1 = acos(ca1);
        dangle_1_dr = x_left / (r2 * sqrt(1 - x_left2 / r2));
        sa0 = y_bottom/r;
        ca0 = sqrt(1-sa0 * sa0);
        // angle_0 = asin(sa0);
        dangle_0_dr = -y_bottom / (r2 * sqrt(1 - y_bottom2 / r2));
    }
    else{
        ca1 = x_left/r;
        sa1 = sqrt(1-ca1*ca1);
        // angle_1 = acos(ca1);
        dangle_1_dr = x_left / (r2 * sqrt(1 - x_left2 / r2));
        ca0 = x_right/r;
        sa0 = sqrt(1-ca0*ca0);
        // angle_0 = acos(ca0);
        dangle_0_dr = x_right / (r2 * sqrt(1 - x_right2 / r2));
    }

    double a = - r / dx * (2 * qx - 1);
    double b = x / dx * (2 * qx - 1);
    double c = - r / dx * ( 2 *qy - 1);
    double d = y / dx * (2 * qy - 1);

    double dadr = - 1/ dx * (2*qx -1);
    double dcdr = -1/dx * (2 * qy - 1);

    double terms [] = {dangle_1_dr, - dangle_0_dr, b * dangle_1_dr, - b * dangle_0_dr,
                       d * dangle_1_dr, - d * dangle_0_dr,
                       b * d * dangle_1_dr, - b * d * dangle_0_dr,
                       a * ca1 * dangle_1_dr, - a * ca0 * dangle_0_dr,
                       d * a * ca1 * dangle_1_dr, - d * a * ca0 * dangle_0_dr,
                       dadr * sa1, -dadr *sa0, d * dadr * sa1, -d * dadr * sa0,
                       -dcdr * ca1, dcdr * ca0, -dcdr * b * ca1, dcdr * b * ca0,
                       c * dangle_1_dr * sa1, - c * dangle_0_dr * sa0,
                       c * b * dangle_1_dr * sa1, - c* b*dangle_0_dr * sa0,
                       -dadr * c / 2 * ca1 * ca1, dadr * c/2 * ca0 * ca0,
                       -dcdr * a /2 *ca1 * ca1, dcdr * a/2 * ca0 * ca0,
                       a * c * ca1 * sa1 * dangle_1_dr, - a * c* ca0 * sa0 * dangle_0_dr
    };

    double kahans = 0.;
    double kahanc = 0.;
    double kahany;
    double kahant;
    double kahanv;
    for (int i = 0; i<30;i ++){
        kahanv = terms[i];
        kahany = kahanv - kahanc;
        kahant = kahans + kahany;
        kahanc = (kahant - kahans) - kahany;
        kahans = kahant;
    }
    return kahans;
}

extern "C" __global__
void calculate_element(double* output, int* indices, int nx, int nt_pixel, double detx,
                       double dety, double dl, double x_0, double dx) {
    int pixel;
    int nt_offset;
    int i_x;
    int i_y;
    double x;
    double y;
    int nt;
    double R;
    double r;

    unsigned int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned int ntid = (gridDim.x * blockDim.x);
    int size = ((nx * nx) * nt_pixel);
    double weight = 0.0;

    // Stride through all required calculations.
    for (int i = tid; i < size; i += ntid) {
        pixel = ((int)(i / nt_pixel));

        // Each loop is a pixel and a offset.
        nt_offset = i % nt_pixel;
        i_x = pixel % nx;
        i_y = ((int)(pixel / nx));

        // Convert index into position. Make detector at (0, 0).
        x = ((x_0 + (dx * (double)(i_x))) - detx);
        y = ((x_0 + (dx * (double)(i_y))) - dety);

        // Include negative offsets.
        nt = (nt_offset - (int)(nt_pixel/ 2));

        // Calculate the key parameters of the system: distance and angle subtended.
        R = (sqrt(((x*x) + (y*y))));

        // Calculate the radius of this time sample
        r = floor(R/dl) * dl + dl * nt;
        // Probably overkill, but this does Kahan summation to try and minimise the round off error.
        // This part can probably be eliminated, it's more important in the sector integrate function.
        // I'm too scared to do this lol ^.
        // Thanks to this: if desired in future, this can *probably* be changed to floats instead of doubles if speed/memory are a huge issue, but I think it should be avoided if possible. Maybe you could convert the final matrices to floats if you really are short of memory.
        double kahanc = 0.;
        weight = 0.;
        double kahans = 0.;
        double kahany;
        double kahant;
        double kahanv;

        for (int qx=0; qx <= 1; qx ++){
            for (int qy = 0; qy<=1; qy++){
                kahanv = sector_integrate(x, y, dx, r, qx, qy);
                kahany = kahanv - kahanc;
                kahant = kahans + kahany;
                kahanc = (kahant - kahans) - kahany;
                kahans = kahant;
                weight += kahanv;
            }
        }

        output[i] = weight;
        indices[i] = (int)(r/dl);
    }
}
