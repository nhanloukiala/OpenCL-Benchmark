#define VEXCL_SHOW_KERNELS // define this macro before VexCL header inclusion to view output kernels

#include <vexcl/vexcl.hpp>
typedef double real;
#include <iostream>
#include <vector>
#include <cstdlib>

void gpuConjugateGradient(const std::vector<size_t> &row,
                          const std::vector<size_t> &col,
                          const std::vector<real> &val,
                          const std::vector<real> &rhs,
                          std::vector<real> &x) {
    /*
     Initialize the OpenCL context
     */
    vex::Context oclCtx(vex::Filter::Type(CL_DEVICE_TYPE_GPU) &&
                        vex::Filter::DoublePrecision);

    size_t n = x.size();
    vex::SpMat<real> A(oclCtx, n, n, row.data(), col.data(), val.data());
    vex::vector<real> f(oclCtx, rhs);
    vex::vector<real> u(oclCtx, x);
    vex::vector<real> r(oclCtx, n);
    vex::vector<real> p(oclCtx, n);
    vex::vector<real> q(oclCtx, n);

    vex::Reductor<real,vex::MAX> max(oclCtx);
    vex::Reductor<real,vex::SUM> sum(oclCtx);

    /*
     Solve the equation Au = f with the "conjugate gradient" method
     See http://en.wikipedia.org/wiki/Conjugate_gradient_method
     */  
    real rho1, rho2;
    r = f - A * u;
    
    for(uint iter = 0; max(fabs(r)) > 1e-8 && iter < n; iter++) {
        rho1 = sum(r * r);
        if(iter == 0 ) {
          p = r;
        } else { 
          real beta = rho1 / rho2;
          p = r + beta * p;
        }

        q = A * p;

        real alpha = rho1 / sum(p * q);

        u += alpha * p;
        r -= alpha * q;

        rho2 = rho1;
    }

    using namespace vex;
    vex::copy(u, x);
}

int main(int argc, char** argv) {
    const unsigned int N = 64;
    size_t rows_[] = {0,3,5,7,8,9,12};
    size_t cols_[] = {0,6,7,2,7,2,3,4,5,0,6,7,2,8};
    real  vals_[] = {2.2,7.1,3.3,8.5,6.2,1.7,6.6,4.5,9.2,2.9,1.3,4.2,3.7,9.8};

    std::vector<size_t> rows(rows_, rows_ + sizeof(rows_)/sizeof(int));
    std::vector<size_t> cols(cols_, cols_ + sizeof(cols_)/sizeof(int)); 
    std::vector<real> val(vals_, vals_ + sizeof(vals_)/sizeof(real));
    std::vector<real> rhs(N);
    std::vector<real> x(N);

    gpuConjugateGradient(rows, cols, val, rhs, x);

}


