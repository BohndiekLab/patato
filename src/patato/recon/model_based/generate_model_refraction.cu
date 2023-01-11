
double fn_x(double x_0, double y_0, double c_0, double c_1, double x, double y){
    // Derivative of time function
    return x_0/(c_0 * sqrt(pow(x_0, 2.) + pow(y_0, 2.))) + (x_0 - x)/(c_1 * sqrt(pow(x_0 - x,2.) + pow(y_0 - y, 2.)));
}

double fprime(double x_0, double y_0, double c_0, double c_1, double x, double y){
    // Second derivative of time function
    return -x_0*x_0/(c_0 * pow(x_0*x_0+ y_0*y_0, 1.5)) + 1/(c_0 * sqrt(x_0*x_0 + y_0*y_0)) + 1/(c_1 * sqrt(pow(x_0 - x, 2.) + pow(y_0 - y, 2.))) - pow(x_0 - x, 2.)/(c_1 * pow(pow(x_0 - x, 2.) + pow(y_0 - y, 2.),1.5));
}

double get_refraction_point(double x, double y, double c0, double c1, double y_cutoff){
    double x_0 = x * (y_cutoff)/(y);
    double x_prev;
    double f_prev;
    double f_new=fn_x(x_0, y_cutoff, c0, c1, x, y);
    int diverging = 0;
    for (int i = 0; i<100; i++){
        // Use NR method to find root.
        x_prev = x_0;
        f_prev = f_new;

        x_0 -= f_new/fprime(x_0, y_cutoff, c0, c1, x, y);

        f_new = fn_x(x_0, y_cutoff, c0, c1, x, y);
        if (f_new > f_prev){
            // Diverging:
            diverging = 1;
            x_0 = x * (y_cutoff)/(y);
            break;
        }
        if (fabs(x_0 - x_prev) < 1e-10){
            break;
        }
    }
//    if (diverging == 1){
//        std::cout << x_0 << " " << x << " " << y << " " << c0 << " " << c1 << " " << y_cutoff << std::endl;
//    }
    return x_0;
}


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
                       double dety, double dl_couple, double dl_tissue, double x_0, double dx,
                       double y_couple) {
    int pixel;
    int nt_offset;
    int i_x;
    int i_y;
    double x;
    double y;
    double x_eff;
    double y_eff;

    int nt;
    double R;
    double r;
    int index;
    double dl;

    unsigned int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned int ntid = (gridDim.x * blockDim.x);
    int size = ((nx * nx) * nt_pixel);
    double weight = 0.0;
    double y_cutoff = y_couple - dety;

    // Stride through all required calculations.
    for (int i = tid; i < size; i += ntid) {
        pixel = ((int)(i / nt_pixel));

        // Each loop is a pixel and a offset.
        nt_offset = i % nt_pixel;
        i_x = pixel % nx;
        i_y = ((int)(pixel / nx));

        x = x_0 + (dx * (double)(i_x));
        y = x_0 + (dx * (double)(i_y));

        // Include negative offsets
        nt = (nt_offset - (int)(nt_pixel/ 2));

        // So, we're calculating the weight for the pixel (x, y) and the time point that is nt away from
        // it's middle. Nothing needs changing above.

        if (y > y_couple){
            // pixel in the coupling region, so no need to adjust the path length
            // Convert index into position. Make detector at (0, 0).
            x_eff = x - detx;
            y_eff = y - dety;
            dl = dl_couple;
        }
        else {
            // pixel below the membrane
            x -= detx;
            y -= dety;

            double x_intercept = get_refraction_point(x, y, dl_couple, dl_tissue, y_cutoff);

            double N = sqrt(x_intercept * x_intercept + y_cutoff * y_cutoff)/dl_couple + sqrt((x_intercept - x) * (x_intercept - x) + (y_cutoff-y) * (y_cutoff-y))/dl_tissue;
            double delta_x = (x-x_intercept) / sqrt((x-x_intercept)*(x-x_intercept) + (y-y_cutoff)*(y-y_cutoff));
            double delta_y = (y-y_cutoff) / sqrt((x-x_intercept)*(x-x_intercept) + (y-y_cutoff)*(y-y_cutoff));
            x_eff = delta_x * N * dl_tissue;
            y_eff = delta_y * N * dl_tissue;

            dl = dl_tissue;
        }

        R = sqrt(pow(x_eff,2.) + pow(y_eff,2.));
        // Calculate the radius of this time sample
        r = floor(R/dl) * dl + dl * nt;
        index = (int) round(r / dl);
        // If this ever needs to be changed to float instead of doubles, be really careful...
        weight = 0.;
        double v;

        // Calculate the weight for each quadrant.
        for (int qx=0; qx <= 1; qx ++){
            for (int qy = 0; qy<=1; qy++){
                v = sector_integrate(x_eff, y_eff, dx, r, qx, qy);
                weight += v;
            }
        }

        output[i] = weight;
        indices[i] = index;
    }
}
