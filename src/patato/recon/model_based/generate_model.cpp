#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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

    double s = 0.;
    for (int i = 0; i<30;i ++){
        s += terms[i];
    }
    return s;
}


void calculate_element(py::array_t<double> output_py, py::array_t<int> indices_py, int nx, int nt_pixel, double detx,
                       double dety, double dl, double x_0, double dx) {
    double *output = static_cast<double *>(output_py.request().ptr);
    int *indices = static_cast<int *>(indices_py.request().ptr);

    int pixel;
    int nt_offset;
    int i_x;
    int i_y;
    double x;
    double y;
    int nt;
    double R;
    double r;

//    int size = ((nx * nx) * nt_pixel);
    double weight = 0.0;
    // Stride through all required calculations.
    for (int i = 0; i < nt_pixel * nx * nx; i ++) {
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
        R = sqrt(((x*x) + (y*y)));

        // Calculate the radius of this time sample
        r = floor(R/dl) * dl + dl * nt;

        // If this ever needs to be changed to float instead of doubles, be really careful...
        weight = 0.;
        double v;

        for (int qx=0; qx <= 1; qx ++){
            for (int qy = 0; qy<=1; qy++){
                v = sector_integrate(x, y, dx, r, qx, qy);
                weight += v;
            }
        }
        output[i] = weight;
        indices[i] = (int)round(r/dl);
    }
}
//
//int main(){
//    // ~ 2 minutes to calculate all 256 detector elements. ~ halved by not using Kahan sum.
//    // Looks like accuracy probably not affected. - Need to confirm.
//
//    // Note the way that I was doing it before (four separate terms then adding), looks like it definitely did lose precision,
//    // but the Kahan summation here is unnecessary at double precision. Consider adding some sort of pairwise adding if
//    // wanting to run at single precision.
//
//    auto start = std::chrono::steady_clock::now();
//
//    int ndet = 256;
//    int nx = 512;
//    int nt_pixel = 7;
//    double* output = new double[nx*nx*nt_pixel*ndet];
//    int* indices = new int[nx*nx*nt_pixel*ndet];
//    double dl = 1540./4e7;
//    double x_0 = -0.025/2;
//    double dx = 0.025/333;
//
//    std::vector<std::thread> threads {};
//
//    for (int i=0; i<ndet; i++){
//        std::cout << i << std::endl;
//        double detx = 0.0405 * cos(i * M_PI * 3 / 2 / ndet);
//        double dety = 0.0405 * sin(i * M_PI * 3 / 2 / ndet);
//
//        threads.push_back(std::thread (calculate_element, output + i*nx*nx*nt_pixel, indices + i*nx*nx*nt_pixel, nx, nt_pixel,
//                    detx, dety, dl, x_0, dx));
//    }
//    for (int i=0; i<threads.size(); i++){
//        threads[i].join();
//    }
//    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
//    std::cout << time.count() << " ms" << std::endl;
//    return 0;
//}

PYBIND11_MODULE(generate_model, m) {
    m.def("calculate_element", &calculate_element, R"pbdoc(Docs here)pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
