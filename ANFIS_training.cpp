/* ANFIStraining.cpp */

//-------------------------------| 
//          LIBRARIES            |
//-------------------------------| 
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


//-------------------------------| 
//    FUNCTION DECLARATION       |
//-------------------------------| 


void mapping(double minIn, double maxIn, double minOut, double maxOut);
double h(double t);
double r(double t);
void fill(double* arr, int n, int m, double val);
double sign(double x);
double map(double input, double minIn, double maxIn, double minOut, double maxOut);

double u1(double t, double amp1);
double u2(double t, double amp2);
double u3(double t, double amp3);

double gauss(double x, double a, double c);
double sigmoid(double x, double a, double c);
double invSigmoid(double x, double a, double c);
double dGauss_da(double x, double a, double c);
double dGauss_dc(double x, double a, double c);
double dinvSigmoid_da(double x, double a, double c);
double dinvSigmoid_dc(double x, double a, double c);


//-------------------------------| 
//            KERNELS            |
//-------------------------------| 



//-------------------------------| 
//       GLOBAL VARIABLES        |
//-------------------------------| 
const double pi = 3.14159265359;
const int mInputs = 3;
const int mStates = 6;
const int nData = 5500;

const int nInputs = 4;
const int nFuzzy = 5;
const int nRules = 625;

double tt[nData];
double UT[mInputs][nData];
double U[mInputs][nData];
double X[mStates][nData];
double Y[mStates][nData];
double Vearth[3][nData];
double Xearth[3][nData];
double lambda[3][3];
double dX[mStates][nData];
double Vtot[nData];
double Wtot[nData];
double G[mStates][nData];
double A[mStates][nData];
double Fd[mStates][nData];
double P[mStates][nData];

double uuu[mInputs][nData];
double xxx[mStates][nData];
double yyy[mStates][nData];

double OUTPUT[nData];
double INPUT1[nData];
double INPUT2[nData];
double INPUT3[nData];
double INPUT4[nData];

double O5[nData];
double En[nData];
double muIn[nFuzzy][nInputs];
double w[nRules];
double wn[nRules];
double fi[nRules];
double fi_wn[nRules];
double sumW;

double dJn_dO5[nData];
double dJn_dO2[nRules];
double dO5_dfi[nRules];
double dO5_dO2[nRules];
double dO2_dO1[nRules][nFuzzy * nInputs];

double dfi_da[nRules];
double dfi_dc[nRules];
double dmu_daIn[nFuzzy][nInputs];
double dmu_dcIn[nFuzzy][nInputs];
double dJn_daOut[nRules];
double dJn_dcOut[nRules];
double dJp_daOut[nRules];
double dJp_dcOut[nRules];
double dJn_dmu[nFuzzy][nInputs];
double dJn_daIn[nFuzzy][nInputs];
double dJn_dcIn[nFuzzy][nInputs];
double dJp_daIn[nFuzzy][nInputs];
double dJp_dcIn[nFuzzy][nInputs];

double aIne[nFuzzy][nInputs];
double cIne[nFuzzy][nInputs];
double aOute[nRules];
double cOute[nRules];
double aInfinal[nFuzzy][nInputs];
double cInfinal[nFuzzy][nInputs];
double aOutfinal[nRules];
double cOutfinal[nRules];



//-------------------------------| 
//            MAIN()             |
//-------------------------------| 
int main()
{
    //----------- Step 0: Start ---------//
    printf("ANFIStraining.exe started...");
    printf("\n\nPiero A. Riva Riquelme, pieroriva@udec.cl");
    printf("\nUndergraduate student at Universidad de Concepcion, Chile");
    printf("\nJanuary of 2021");
    double seconds;
    time_t timer1;
    time_t timer2;
    time(&timer1);

    //----------- Step 1: Process definition ---------//
    printf("\n\nStep 1: Process definition");
    // a) Physical constants
    printf("\n\ta) Physical constants");
    const double rhoAir = 1.205;            // Density of air at NTP (20°C, 1atm)
    const double rhoHe = 0.1664;           // Density of Helium at NTP (20°C, 1atm)
    const double g_acc = 9.80665;           // Acceleration of gravity
    const double deg2rad = pi / 180;          // Degrees to radians conversion 
    const double rad2deg = pow(deg2rad, -1.0); // Radians to degrees conversion

    // b) Vehicle geometry and parameters
    printf("\n\tb) Vehicle geometry and parameters");
    const double blimp_a = 0.9;                                     // Blimp's x length 
    const double blimp_b = 0.45;                                    // Blimp's y length 
    const double blimp_c = 0.45;                                    // Blimp's z length 
    const double blimp_volume = 4.0 * pi * blimp_a * blimp_b * blimp_c / 3.0;    // Blimp's volume 
    const double blimp_area = pow(blimp_volume, 0.6666666666);      // Blimp's area 
    const double blimp_mHe = blimp_volume * rhoHe;                    // Blimp's mass of helium 
    const double blimp_mAir = blimp_volume * rhoAir;                  // Blimp's mass of air
    const double blimp_mass = blimp_mAir - blimp_mHe;               // Blimp's mass (chosen for 0 buoyancy)
    const double blimp_mTotal = blimp_mass + blimp_mHe;               // Blimp's total mass
    const double blimp_dx = blimp_a / 8.0;                              // Blimp's x axis distace from CV to propellers
    const double blimp_dy = blimp_b / 2.0;                              // Blimp's y axis distace from CV to propellers
    const double blimp_dz = blimp_c;                                // Blimp's z axis distace from CV to propellers
    const double blimp_ax = 0.0;                                      // Blimp's x axis distance from center of gravity CG and center of volume CV
    //const double blimp_ay = 0.0;                                      // Blimp's y axis distance from center of gravity CG and center of volume CV
    const double blimp_az = -(0.2 * blimp_mass) * blimp_b / blimp_mTotal;   // Blimp's z axis distance from center of gravity CG and center of volume CV

    // c) Masses and inertias
    printf("\n\tc) Masses and inertias");
    const double blimp_Ix = blimp_mTotal * (blimp_b * blimp_b + blimp_c * blimp_c) / 5.0;
    const double blimp_Iy = blimp_mTotal * (blimp_c * blimp_c + blimp_a * blimp_a) / 5.0;
    const double blimp_Iz = blimp_mTotal * (blimp_a * blimp_a + blimp_b * blimp_b) / 5.0;

    // c.1) Tuckerman fo a prolate ellipsoid
    const double tuckerman_e = sqrt(1.0 - blimp_c * blimp_c / (blimp_a * blimp_a));
    const double tuckerman_alpha = (1.0 - tuckerman_e * tuckerman_e) * (log((1.0 + tuckerman_e) / (1.0 - tuckerman_e)) - 2.0 * tuckerman_e) / (tuckerman_e * tuckerman_e * tuckerman_e);
    const double tuckerman_beta = (1.0 - tuckerman_e * tuckerman_e) * ((tuckerman_e / (1.0 - tuckerman_e * tuckerman_e)) - 0.5 * log((1.0 + tuckerman_e) / (1.0 - tuckerman_e))) / (tuckerman_e * tuckerman_e * tuckerman_e);
    const double tuckerman_gamma = tuckerman_beta;

    const double tuckerman_K1 = blimp_volume * (tuckerman_alpha / (2.0 - tuckerman_alpha));
    const double tuckerman_K2 = blimp_volume * (tuckerman_beta / (2.0 - tuckerman_beta));
    const double tuckerman_K3 = blimp_volume * (tuckerman_gamma / (2.0 - tuckerman_gamma));
    const double tuckerman_K1_ = blimp_volume * blimp_Ix * (pow((blimp_b * blimp_b - blimp_c * blimp_c) / (blimp_b * blimp_b + blimp_c * blimp_c), 2.0) * ((tuckerman_gamma - tuckerman_beta) / (2.0 * ((blimp_b * blimp_b - blimp_c * blimp_c) / (blimp_b * blimp_b + blimp_c * blimp_c)) - (tuckerman_gamma - tuckerman_beta))));
    const double tuckerman_K2_ = blimp_volume * blimp_Iy * (pow((blimp_c * blimp_c - blimp_a * blimp_a) / (blimp_c * blimp_c + blimp_a * blimp_a), 2.0) * ((tuckerman_alpha - tuckerman_gamma) / (2.0 * ((blimp_c * blimp_c - blimp_a * blimp_a) / (blimp_c * blimp_c + blimp_a * blimp_a)) - (tuckerman_alpha - tuckerman_gamma))));
    const double tuckerman_K3_ = blimp_volume * blimp_Iz * (pow((blimp_a * blimp_a - blimp_b * blimp_b) / (blimp_a * blimp_a + blimp_b * blimp_b), 2.0) * ((tuckerman_beta - tuckerman_alpha) / (2.0 * ((blimp_a * blimp_a - blimp_b * blimp_b) / (blimp_a * blimp_a + blimp_b * blimp_b)) - (tuckerman_beta - tuckerman_alpha))));

    // c.2) Virtual masses and inertias
        // Tuckerman
    const double blimp_Xu = -tuckerman_K1 * rhoAir;
    const double blimp_Yv = -tuckerman_K2 * rhoAir;
    const double blimp_Zw = -tuckerman_K3 * rhoAir;
    const double blimp_Lp = 0.0;
    const double blimp_Mq = -tuckerman_K2_ * rhoAir;
    const double blimp_Nr = -tuckerman_K3_ * rhoAir;

    // Gomes
    const double blimp_Mu = 0.0;
    const double blimp_Lv = 0.0;
    const double blimp_Nv = 0.0;
    const double blimp_Mw = 0.0;
    const double blimp_Yp = 0.0;
    const double blimp_Xq = 0.0;
    const double blimp_Zq = 0.0;
    const double blimp_Yr = 0.0;

    // Groups
    const double blimp_mx = blimp_mTotal - blimp_Xu;
    const double blimp_my = blimp_mTotal - blimp_Yv;
    const double blimp_mz = blimp_mTotal - blimp_Zw;
    const double blimp_Jx = blimp_Ix - blimp_Lp;
    const double blimp_Jy = blimp_Iy - blimp_Mq;
    const double blimp_Jz = blimp_Iz - blimp_Nr;
    const double blimp_Jxz = 0.0;


    // d) M matrix
    printf("\n\td) M matrix");
    const double M[6][6] = {
        {blimp_mx                           , 0.0                                   , 0.0                                   , 0.0                                   , blimp_mTotal * blimp_az - blimp_Xq    , 0.0                               },
        {0.0                                , blimp_my                              , 0.0                                   , -blimp_mTotal * blimp_az - blimp_Yp   , 0.0                                   , blimp_mTotal * blimp_ax - blimp_Yr},
        {0.0                                , 0.0                                   , blimp_mz                              , 0.0                                   , -blimp_mTotal * blimp_ax - blimp_Zq   , 0.0                               },
        {0.0                                , -blimp_mTotal * blimp_az - blimp_Lv   , 0.0                                   , blimp_Ix - blimp_Lp                   , 0.0                                   , -blimp_Jxz                        },
        {blimp_mTotal * blimp_az - blimp_Mu , 0.0                                   , -blimp_mTotal * blimp_ax - blimp_Mw   , 0.0                                   , blimp_Iy - blimp_Mq                   , 0.0                               },
        {0.0                                , blimp_mTotal * blimp_ax - blimp_Nv    , 0.0                                   , -blimp_Jxz                            , 0.0                                   , blimp_Iz - blimp_Nr               }
    };

    const double invM[6][6] = {
        {0.916844157881075  , 0.0                   , 0.0               , 0.0                   , 0.287823601986896 , 0.0               },
        {0.0                , 0.666945041828523     , 0.0               , -0.638717492340345    , 0.0               , 0.0               },
        {0.0                , 0.0                   , 0.637872073637673 , 0.0                   , 0.0               , 0.0               },
        {0.0                , -0.638717492340345    , 0.0               , 14.032280169794683    , 0.0               , 0.0               },
        {0.287823601986896  , 0.0                   , 0.0               , 0.0                   , 4.489659394858919 , 0.0               },
        {0.0                , 0.0                   , 0.0               , 0.0                   , 0.0               , 4.399303334727424 }
    };

    /*for (int i = 0; i < mStates; i++) {
       printf("\n");
       for (int j = 0; j < mStates; j++) {
           printf("%.2f ", invM[i][j]);
       }
   } */


   //----------- Step 2: Simulation configuration ---------//
    printf("\n\nStep 2: Simulation configuration");
    // a) Time definition
    printf("\n\ta) Time definition");
    double ti = 10.1;
    double step = 0.1;
    double tf = 560;
    //double tt[nData];
    for (int i = 0; i < nData; i++) {
        tt[i] = ti + step * i;
    }


    // b) Process configuration and declaration
    printf("\n\tb) Process configuration and declaration");
    const double tol = pow(10, -6.0);

    fill((double*)UT, mInputs, nData, 0.0);
    fill((double*)UT + nData, 1, nData, 0.5);
    fill((double*)U, mInputs, nData, 0.0);
    fill((double*)X, mStates, nData, 0.0);
    fill((double*)Y, mStates, nData, 0.0);
    fill((double*)Vearth, 3, nData, 0.0);
    fill((double*)Xearth, 3, nData, 0.0);
    fill((double*)lambda, 3, 3, 0.0);
    fill((double*)dX, mStates, nData, 0.0);
    fill((double*)Vtot, nData, 1, 0.0);
    fill((double*)Wtot, nData, 1, 0.0);
    fill((double*)G, mStates, nData, 0.0);
    fill((double*)A, mStates, nData, 0.0);
    fill((double*)Fd, mStates, nData, 0.0);
    fill((double*)P, mStates, nData, 0.0);

    /* for (int i = 0; i < nData; i++) {
        printf("\n");
        for (int j = 0; j < 3; j++) {
            printf("%.2f ", U[j][i]);
        }
    } */


    // c) Tools definition
    printf("\n\tc) Tools definition");
    // heaviside & ramp

// d) Input generation
    printf("\n\td) Input generation");
    const double amp1 = 0.1;
    const double amp2 = 0.05;
    const double amp3 = 90 * deg2rad;
    for (int i = 0; i < nData; i++) {
        UT[0][i] = (u1(tt[i], amp1)) + (((double)rand() / RAND_MAX) - 0.5) * amp1 * 0.2;
        UT[1][i] = (u2(tt[i], amp2)) + (((double)rand() / RAND_MAX) - 0.5) * amp2 * 0.2;
        UT[2][i] = (u3(tt[i], amp3)) + (((double)rand() / RAND_MAX) - 0.5) * amp3 * 0.2;

        if (UT[0][i] < -amp1) { UT[0][i] = -amp1; }
        if (UT[0][i] > amp1) { UT[0][i] = amp1; }
        if (UT[1][i] < 0.5 - amp2) { UT[1][i] = 0.5 - amp2; }
        if (UT[1][i] > 0.5 + amp2) { UT[1][i] = 0.5 + amp2; }
        if (UT[2][i] < -amp3) { UT[2][i] = -amp3; }
        if (UT[2][i] > amp3) { UT[2][i] = amp3; }
    }

    /*
     for (int i = 0; i < nData; i++) {
         printf("\n");
         for (int j = 0; j < 3; j++) {
             printf("%.5f ", UT[j][i]);
         }
     } */


     //----------- Step 3: Simulation started ---------//
    printf("\n\nStep 3: Simulation started");

    for (int n = 1; n < nData; n++) {
        // a) Dynamics vector, Fd
        double f1 = -blimp_mz * X[2][n - 1] * X[4][n - 1] + blimp_my * X[5][n - 1] * X[1][n - 1] + blimp_mTotal * (blimp_ax * (X[4][n - 1] * X[4][n - 1] + X[5][n - 1] * X[5][n - 1]) - blimp_az * X[5][n - 1] * X[3][n - 1]);
        double f2 = -blimp_mx * X[0][n - 1] * X[5][n - 1] + blimp_mz * X[3][n - 1] * X[2][n - 1] + blimp_mTotal * (-blimp_ax * X[3][n - 1] * X[4][n - 1] - blimp_az * X[5][n - 1] * X[4][n - 1]);
        double f3 = -blimp_my * X[1][n - 1] * X[3][n - 1] + blimp_mx * X[4][n - 1] * X[0][n - 1] + blimp_mTotal * (-blimp_ax * X[5][n - 1] * X[3][n - 1] + blimp_az * (X[4][n - 1] * X[4][n - 1] + X[3][n - 1] * X[3][n - 1]));
        double f4 = -(blimp_Jz - blimp_Jy) * X[5][n - 1] * X[4][n - 1] + blimp_Jxz * X[3][n - 1] * X[4][n - 1] + blimp_mTotal * blimp_az * (X[0][n - 1] * X[5][n - 1] - X[3][n - 1] * X[2][n - 1]);
        double f5 = -(blimp_Jx - blimp_Jz) * X[3][n - 1] * X[5][n - 1] + blimp_Jxz * (X[5][n - 1] * X[5][n - 1] - X[3][n - 1] * X[3][n - 1]) + blimp_mTotal * (blimp_ax * (X[1][n - 1] * X[3][n - 1] - X[4][n - 1] * X[0][n - 1]) - blimp_az * (X[2][n - 1] * X[4][n - 1] - X[5][n - 1] * X[1][n - 1]));
        double f6 = -(blimp_Jy - blimp_Jx) * X[4][n - 1] * X[3][n - 1] - blimp_Jxz * X[4][n - 1] * X[5][n - 1] + blimp_mTotal * (-blimp_ax * (X[0][n - 1] * X[5][n - 1] - X[3][n - 1] * X[2][n - 1]));
        Fd[0][n - 1] = f1;
        Fd[1][n - 1] = f2;
        Fd[2][n - 1] = f3;
        Fd[3][n - 1] = f4;
        Fd[4][n - 1] = f5;
        Fd[5][n - 1] = f6;

        // b) Propulsion vector, P
        U[0][n - 1] = UT[0][n - 1] * UT[1][n - 1];            // Alpha* Tmax
        U[1][n - 1] = UT[0][n - 1] * (1.0 - UT[1][n - 1]);      // (1 - Alpha)* Tmax
        U[2][n - 1] = UT[2][n - 1];

        double P1 = (U[0][n - 1] + U[1][n - 1]) * cos(U[2][n - 1]);
        double P2 = 0;
        double P3 = -(U[0][n - 1] + U[1][n - 1]) * sin(U[2][n - 1]);
        double P4 = (U[1][n - 1] - U[0][n - 1]) * sin(U[2][n - 1]) * blimp_dy;
        double P5 = (U[0][n - 1] + U[1][n - 1]) * (blimp_dz * cos(U[2][n - 1]) - blimp_dx * sin(U[2][n - 1]));
        double P6 = (U[1][n - 1] - U[0][n - 1]) * cos(U[2][n - 1]) * blimp_dy;
        P[0][n - 1] = P1;
        P[1][n - 1] = P2;
        P[2][n - 1] = P3;
        P[3][n - 1] = P4;
        P[4][n - 1] = P5;
        P[5][n - 1] = P6;

        // c) Aerodynamic force vector, A
        Vtot[n - 1] = pow(X[0][n - 1] * X[0][n - 1] + X[1][n - 1] * X[1][n - 1] + X[2][n - 1] * X[2][n - 1], 0.5);
        Wtot[n - 1] = pow(X[3][n - 1] * X[3][n - 1] + X[4][n - 1] * X[4][n - 1] + X[5][n - 1] * X[5][n - 1], 0.5);

        double CD = 0.9;
        double CY = 0.9;
        double CL = 0.9;
        double Cl = 0.9;
        double Cm = 0.9;
        double Cn = 0.9;

        double coefB1 = 0.5 * rhoAir * X[0][n - 1] * X[0][n - 1] * sign(X[0][n - 1]) * blimp_area;
        double coefB2 = 0.5 * rhoAir * X[1][n - 1] * X[1][n - 1] * sign(X[1][n - 1]) * blimp_area;
        double coefB3 = 0.5 * rhoAir * X[2][n - 1] * X[2][n - 1] * sign(X[2][n - 1]) * blimp_area;
        double coefB4 = 0.5 * rhoAir * X[3][n - 1] * X[3][n - 1] * sign(X[3][n - 1]) * blimp_volume;
        double coefB5 = 0.5 * rhoAir * X[4][n - 1] * X[4][n - 1] * sign(X[4][n - 1]) * blimp_volume;
        double coefB6 = 0.5 * rhoAir * X[5][n - 1] * X[5][n - 1] * sign(X[5][n - 1]) * blimp_volume;

        double A1 = -CD * coefB1;
        double A2 = -CY * coefB2;
        double A3 = -CL * coefB3;
        double A4 = -Cl * coefB4;
        double A5 = -Cm * coefB5;
        double A6 = -Cn * coefB6;

        A[0][n - 1] = A1;
        A[1][n - 1] = A2;
        A[2][n - 1] = A3;
        A[3][n - 1] = A4;
        A[4][n - 1] = A5;
        A[5][n - 1] = A6;

        // d) Gravitational force vector, G
        lambda[0][0] = cos(Y[4][n - 1]) * cos(Y[5][n - 1]);
        lambda[0][1] = cos(Y[4][n - 1]) * sin(Y[5][n - 1]);
        lambda[0][2] = sin(Y[4][n - 1]);
        lambda[1][0] = (-cos(Y[3][n - 1]) * sin(Y[5][n - 1]) + sin(Y[3][n - 1]) * sin(Y[4][n - 1]) * cos(Y[5][n - 1]));
        lambda[1][1] = (cos(Y[3][n - 1]) * cos(Y[5][n - 1]) + sin(Y[3][n - 1]) * sin(Y[4][n - 1]) * sin(Y[5][n - 1]));
        lambda[1][2] = sin(Y[3][n - 1]) * cos(Y[4][n - 1]);
        lambda[2][0] = (sin(Y[3][n - 1]) * sin(Y[5][n - 1]) + cos(Y[3][n - 1]) * sin(Y[4][n - 1]) * cos(Y[5][n - 1]));
        lambda[2][1] = (-sin(Y[3][n - 1]) * cos(Y[5][n - 1]) + cos(Y[3][n - 1]) * sin(Y[4][n - 1]) * sin(Y[5][n - 1]));
        lambda[2][2] = cos(Y[3][n - 1]) * cos(Y[4][n - 1]);

        double B = rhoAir * g_acc * blimp_volume;
        double W = blimp_mTotal * g_acc;

        double G1 = lambda[2][0] * (W - B);
        double G2 = lambda[2][1] * (W - B);
        double G3 = lambda[2][2] * (W - B);
        double G4 = -lambda[2][1] * blimp_az * W;
        double G5 = (lambda[2][0] * blimp_az - lambda[2][2] * blimp_ax) * W;
        double G6 = lambda[2][1] * blimp_ax * W;

        G[0][n - 1] = G1;
        G[1][n - 1] = G2;
        G[2][n - 1] = G3;
        G[3][n - 1] = G4;
        G[4][n - 1] = G5;
        G[5][n - 1] = G6;

        // e) Differential equation
        double aux_differential_equation[mStates];
        for (int i = 0; i < mStates; i++) {
            aux_differential_equation[i] = P[i][n - 1] + Fd[i][n - 1] + A[i][n - 1] + G[i][n - 1];
        }

        for (int i = 0; i < mStates; i++) {
            for (int j = 0; j < mStates; j++) {
                dX[i][n - 1] = dX[i][n - 1] + invM[i][j] * aux_differential_equation[j];
            }
        }

        // f) Integrate differential equation
        for (int i = 0; i < mStates; i++) {
            for (int j = 0; j < n; j++) {
                X[i][n] = X[i][n] + dX[i][j];
            }
            X[i][n] = X[i][n] + (dX[i][n - 1] - dX[i][0]) * 0.5;
            X[i][n] = X[i][n] * step;
        }

        // g) Transform linear velocities to earth axes to obtain Vnort, Veast, Vup
            //

        // h) Calculate vehicle position in terms of displacements in the north, east and vertical directions 
        for (int i = 0; i < mStates; i++) {
            for (int j = 0; j < n; j++) {
                Y[i][n] = Y[i][n] + X[i][j];
            }
            Y[i][n] = Y[i][n] + (X[i][n - 1] - X[i][0]) * 0.5;
            Y[i][n] = Y[i][n] * step;                           // XYZ aligned with NEU
        }
    }

    //----------- Step 4: Simulation results ---------//
    printf("\n\nStep 4: Simulation results");
    printf("\n\ta) Writing to BlimpSim_Results.m file...");

    FILE* blimpFile;
    blimpFile = fopen("BlimpSim_Results.m", "w");
    if (blimpFile != NULL) {
        for (int j = 1; j <= nData; j++) {
            fprintf(blimpFile, "\ntt_cpp(%d)\t= %03.15f;", j, tt[j - 1]);
        }
        for (int i = 1; i <= mInputs; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nU_cpp(%d,%d)\t= %03.15f;", i, j, U[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mInputs; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nUT_cpp(%d,%d)\t= %03.15f;", i, j, UT[i - 1][j - 1]);
            }
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(blimpFile, "\nVtot_cpp(%d)\t= %03.15f;", j, Vtot[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(blimpFile, "\nwTot_cpp(%d)\t= %03.15f;", j, Wtot[j - 1]);
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nX_cpp(%d,%d)\t= %03.15f;", i, j, X[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nY_cpp(%d,%d)\t= %03.15f;", i, j, Y[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nFd_cpp(%d,%d)\t= %03.15f;", i, j, Fd[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nG_cpp(%d,%d)\t= %03.15f;", i, j, G[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nA_cpp(%d,%d)\t= %03.15f;", i, j, A[i - 1][j - 1]);
            }
        }
        for (int i = 1; i <= mStates; i++) {
            for (int j = 1; j <= nData; j++) {
                fprintf(blimpFile, "\nP_cpp(%d,%d)\t= %03.15f;", i, j, P[i - 1][j - 1]);
            }
        }
        printf("\n\t¡Success!");
        fclose(blimpFile);
    }



    //----------- Step 5: Scaling to fuzzy sets ---------//
    printf("\n\nStep 5: Scaling to fuzzy sets");

    // a) Parameters
    printf("\n\ta) Initializing parameters");
    const double minOut = 10.0;     // Porcentaje
    const double maxOut = 90.0;     // Porcentaje

    const double minUT1 = -0.1;     // Fuerza(N)
    const double maxUT1 = 0.1;      // Fuerza(N)
    const double minUT2 = 0.45;     // Adimensional
    const double maxUT2 = 0.55;     // Adimensional
    const double minUT3 = -pi / 2;  // Ángulo(rad)
    const double maxUT3 = pi / 2;   // Ángulo(rad)

    const double minX1 = -0.4;      // Velocidad(m / s)
    const double maxX1 = 0.4;       // Velocidad(m / s)
    const double minX2 = -1.0;      // Velocidad(m / s)
    const double maxX2 = 1.0;       // Velocidad(m / s)
    const double minX3 = -1.0;      // Velocidad(m / s)
    const double maxX3 = 1.0;       // Velocidad(m / s)
    const double minX4 = -pi;       // Velocidad angular(rad / s)
    const double maxX4 = pi;        // Velocidad angular(rad / s)
    const double minX5 = -pi;       // Velocidad angular(rad / s)
    const double maxX5 = pi;        // Velocidad angular(rad / s)
    const double minX6 = -pi;       // Velocidad angular(rad / s)
    const double maxX6 = pi;        // Velocidad angular(rad / s)

    const double minY1 = -100;      // Posición(m)
    const double maxY1 = 100;       // Posición(m)
    const double minY2 = -100;      // Posición(m)
    const double maxY2 = 100;       // Posición(m)
    const double minY3 = -100;      // Posición(m)
    const double maxY3 = 100;       // Posición(m)
    const double minY4 = -pi;       // Orientación(rad)
    const double maxY4 = pi;        // Orientación(rad)
    const double minY5 = -pi;       // Orientación(rad)
    const double maxY5 = pi;        // Orientación(rad)
    const double minY6 = -pi;       // Orientación(rad)
    const double maxY6 = pi;        // Orientación(rad)

    // b) Mappping
    printf("\n\tb) Mapping...");
    for (int i = 0; i < nData; i++) {
        uuu[0][i] = map(UT[0][i], minUT1, maxUT1, minOut, maxOut);
        uuu[1][i] = map(UT[1][i], minUT2, maxUT2, minOut, maxOut);
        uuu[2][i] = map(UT[2][i], minUT3, maxUT3, minOut, maxOut);

        xxx[0][i] = map(X[0][i], minX1, maxX1, minOut, maxOut);
        xxx[1][i] = map(X[1][i], minX2, maxX2, minOut, maxOut);
        xxx[2][i] = map(X[2][i], minX3, maxX3, minOut, maxOut);
        xxx[3][i] = map(X[3][i], minX4, maxX4, minOut, maxOut);
        xxx[4][i] = map(X[4][i], minX5, maxX5, minOut, maxOut);
        xxx[5][i] = map(X[5][i], minX6, maxX6, minOut, maxOut);

        yyy[0][i] = map(Y[0][i], minY1, maxY1, minOut, maxOut);
        yyy[1][i] = map(Y[1][i], minY2, maxY2, minOut, maxOut);
        yyy[2][i] = map(Y[2][i], minY3, maxY3, minOut, maxOut);
        yyy[3][i] = map(Y[3][i], minY4, maxY4, minOut, maxOut);
        yyy[4][i] = map(Y[4][i], minY5, maxY5, minOut, maxOut);
        yyy[5][i] = map(Y[5][i], minY6, maxY6, minOut, maxOut);
    }

    //----------- Step 6: Training configuration and space generation ---------//
    printf("\n\nStep 6: Training configuration and space generation");
    // a) Training configuration
    printf("\n\ta) Time definition");
    const int maxEpochs = 1000;                   // The more the merrier
    //int nInputs = 4;                          // Number of inputs (needs ANFIS configuring)
    //int nFuzzy = 5;                           // Number of MFs per input
    //int nOutputs = 1;                         // Not changeable
    //int nRules = (int)pow(nFuzzy,nInputs);    // Number of rules
    double K = 0.02;                            // Initial K
    const double maxK = 0.25;                     // Maximum K
    const double growthRate = 0.1;                // Growth of K
    const int backwards = 1;                      // Prediction gap
    const double aIno = 15;                       // Initial "a" parameter of premise MFs
    const double aOuto = 0.01;                    // Initial "a" parameter of consequent MFs

    // b) I/O setup
    printf("\n\tb) Process configuration and declaration");
    for (int i = 0; i < nData; i++) {
        OUTPUT[i] = xxx[0][i];
        INPUT1[i] = uuu[0][i];
        INPUT2[i] = uuu[1][i];
        INPUT3[i] = uuu[2][i];
        INPUT4[i] = xxx[0][i];
    }

    // c) Tools definition
    printf("\n\tc) Tools definition");
    // gauss and sigmoids at the bottom

// d) Workspace generation
    printf("\n\td) Workspace generation");
    // d.1) Initial fuzzy parameteres
    printf("\n\t\td.1) Initial fuzzy parameters");

    for (int i = 0; i < nInputs; i++) {
        for (int j = 0; j < nFuzzy; j++) {
            aIne[j][i] = aIno;
            cIne[j][i] = (100.0 / (nFuzzy - 1.0)) * (j);
        }
    }

    for (int j = 0; j < nRules; j++) {
        aOute[j] = aOuto;
        cOute[j] = (100.0 / (nRules - 1.0)) * (j);
    }

    // d.2) Training workspace
    printf("\n\t\td.2) Training workspace");
    double APE[maxEpochs];
    double APEmin = 100000;
    int epochFlag = 1;
    double XX[nInputs];
    fill((double*)O5, nData, 1, 0.0);
    fill((double*)En, nData, 1, 0.0);
    fill((double*)muIn, nFuzzy, nInputs, 0.0);
    fill((double*)w, nRules, 1, 0.0);
    fill((double*)wn, nRules, 1, 0.0);
    fill((double*)fi, nRules, 1, 0.0);
    fill((double*)fi_wn, nRules, 1, 0.0);
    sumW = 0;

    fill((double*)dJn_dO5, nData, 1, 0.0);
    fill((double*)dJn_dO2, nRules, 1, 0.0);
    fill((double*)dO5_dfi, nRules, 1, 0.0);
    fill((double*)dO5_dO2, nRules, 1, 0.0);
    fill((double*)dO2_dO1, nRules, nFuzzy * nInputs, 0.0);

    fill((double*)dfi_da, nRules, 1, 0.0);
    fill((double*)dfi_dc, nRules, 1, 0.0);
    fill((double*)dmu_daIn, nFuzzy, nInputs, 0.0);
    fill((double*)dmu_dcIn, nFuzzy, nInputs, 0.0);
    fill((double*)dJn_daOut, nRules, 1, 0.0);
    fill((double*)dJn_dcOut, nRules, 1, 0.0);
    fill((double*)dJp_daOut, nRules, 1, 0.0);
    fill((double*)dJp_dcOut, nRules, 1, 0.0);
    fill((double*)dJn_dmu, nFuzzy, nInputs, 0.0);
    fill((double*)dJn_daIn, nFuzzy, nInputs, 0.0);
    fill((double*)dJn_dcIn, nFuzzy, nInputs, 0.0);
    fill((double*)dJp_daIn, nFuzzy, nInputs, 0.0);
    fill((double*)dJp_dcIn, nFuzzy, nInputs, 0.0);

    // d.3) Index table
    printf("\n\t\td.2) Index table");
    static int indexTable[nRules][nInputs];
    for (int k = 1; k <= nInputs; k++) {
        int l = 1;
        for (int j = 1; j <= nRules; j = j + (int)pow(nFuzzy, long long(k) - 1)) {
            for (int i = 1; i <= (int)pow(nFuzzy, (long long(k) - 1)); i++) {
                indexTable[j + i - 2][nInputs - k] = l;
            }
            l = l + 1;
            if (l > nFuzzy) {
                l = 1;
            }
        }
    }

    /*      for(int i=1; i<=nRules;i++){
                printf("\n");
            for(int j=1;j<=nInputs;j++){
                printf(" %d", indexTable[i-1][j-1]);
            }
        }*/

        // e) Plotting step 6
    printf("\n\te) Plotting step 6");



    //----------- Step 7: ANFIS offline training ---------//
    clock_t epochTime;
    printf("\n\nStep 7: ANFIS offline training");
    for (int g = 1; g < maxEpochs; g++) {
        epochTime = clock();
        // Zeroing the derivatives
        for (int i = 0; i < nRules; i++) {
            dJp_daOut[i] = 0.0;
            dJp_dcOut[i] = 0.0;
        }
        for (int i = 0; i < nFuzzy; i++) {
            for (int j = 0; j < nInputs; j++) {
                dJp_daIn[i][j] = 0.0;
                dJp_dcIn[i][j] = 0.0;
            }
        }

        // Sweep through nData starts
        for (int i = 1 + backwards; i < nData; i++) {
            // a) ANFIS
            for (int ANFIS = 0; ANFIS < 1; ANFIS++) {
                // Prelayer
                XX[0] = INPUT1[i - backwards];
                XX[1] = INPUT2[i - backwards];
                XX[2] = INPUT3[i - backwards];
                XX[3] = INPUT4[i - backwards];
                // Layer 1: Input fuzzyfication
                for (int k = 0; k < nInputs; k++) {
                    for (int j = 0; j < nFuzzy; j++) {
                        muIn[j][k] = gauss(XX[k], aIne[j][k], cIne[j][k]);
                        if (muIn[j][k] < tol) {
                            muIn[j][k] = tol;
                        }
                        if (muIn[j][k] > (1.0 - tol)) {
                            muIn[j][k] = 1.0 - tol;
                        }
                    }
                }
                // Layer 2: Calculation weights
                for (int j = 0; j < nRules; j++) {
                    w[j] = 1.0;
                    for (int k = 0; k < nInputs; k++) {
                        w[j] = w[j] * muIn[indexTable[j][k] - 1][k];
                    }
                }
                // Layer 3: Normalizing
                sumW = 0.0;
                for (int j = 0; j < nRules; j++) {
                    sumW = sumW + w[j];
                }
                if (sqrt(sumW * sumW) < tol) {
                    sumW = tol;
                }
                for (int j = 0; j < nRules; j++) {
                    wn[j] = w[j] / sumW;
                }
                // Layer 4: Calculation of wn*fi
                for (int j = 0; j < nRules; j++) {
                    fi[j] = invSigmoid(w[j], aOute[j], cOute[j]);
                    fi_wn[j] = fi[j] * wn[j];
                }
                // Layer 5: Calculating output
                double f = 0;
                for (int j = 0; j < nRules; j++) {
                    f = f + fi_wn[j];
                }
                O5[i] = f;
            }

            // b) ANFIS error measure
            En[i] = OUTPUT[i] - O5[i];
            // dJn/dO5
            dJn_dO5[i] = -2.0 * En[i];

            // c) Gradient descent for consequent parameters
            for (int j = 0; j < nRules; j++) {
                // dO5_dfi
                dO5_dfi[j] = wn[j];
                // dfi_dalpha
                dfi_da[j] = -log(1.0 / w[j] - 1.0); //dinvSigmoid_da(w(j),aOute(j,1),cOute(j,1));
                dfi_dc[j] = 1.0;
                // dJn_dalpha
                dJn_daOut[j] = dJn_dO5[i] * dO5_dfi[j] * dfi_da[j];
                dJn_dcOut[j] = dJn_dO5[i] * dO5_dfi[j] * dfi_dc[j];
                // SUMA
                dJp_daOut[j] = dJp_daOut[j] + dJn_daOut[j];
                dJp_dcOut[j] = dJp_dcOut[j] + dJn_dcOut[j];
            }

            // d) Gradient descent for premise parameters
            // dO5/dO2 = (fi-ye)/sum(w)
            for (int j = 0; j < nRules; j++) {
                dO5_dO2[j] = (fi[j] - O5[i]) / sumW;
            }
            // dO2_dO1 matrix
            for (int e = 1; e <= nInputs; e++) {
                for (int k = 1; k <= nFuzzy; k++) {
                    for (int j = 1; j <= nRules; j++) {
                        if (muIn[k - 1][e - 1] == muIn[indexTable[j - 1][e - 1] - 1][e - 1]) {
                            dO2_dO1[j - 1][(e - 1) * nFuzzy + (k - 1)] = 1.0;
                            for (int p = 1; p <= nInputs; p++) {
                                if (muIn[k - 1][e - 1] != muIn[indexTable[j - 1][p - 1] - 1][p - 1]) {
                                    dO2_dO1[j - 1][(e - 1) * nFuzzy + (k - 1)] = dO2_dO1[j - 1][(e - 1) * nFuzzy + (k - 1)] * muIn[indexTable[j - 1][p - 1] - 1][p - 1];
                                }
                            }
                        }
                        else {
                            dO2_dO1[j - 1][(e - 1) * nFuzzy + (k - 1)] = 0.0;
                        }
                    }
                }
            }

            // dJn_dO2
            for (int j = 0; j < nRules; j++) {
                dJn_dO2[j] = dJn_dO5[i] * dO5_dO2[j];
            }

            // Chain rule
            for (int k = 1; k <= nInputs; k++) {
                for (int j = 1; j <= nFuzzy; j++) {
                    // dJn_dO1
                    dJn_dmu[j - 1][k - 1] = 0.0;
                    for (int p = 1; p <= nRules; p++) {
                        dJn_dmu[j - 1][k - 1] = dJn_dmu[j - 1][k - 1] + dJn_dO2[p - 1] * dO2_dO1[p - 1][(j - 1) + (k - 1) * nFuzzy];
                    }
                    // dO1_dalpha
                    dmu_daIn[j - 1][k - 1] = dGauss_da(XX[k - 1], aIne[j - 1][k - 1], cIne[j - 1][k - 1]);
                    dmu_dcIn[j - 1][k - 1] = dGauss_dc(XX[k - 1], aIne[j - 1][k - 1], cIne[j - 1][k - 1]);
                    // dJn_dalpha
                    dJn_daIn[j - 1][k - 1] = dJn_dmu[j - 1][k - 1] * dmu_daIn[j - 1][k - 1];
                    dJn_dcIn[j - 1][k - 1] = dJn_dmu[j - 1][k - 1] * dmu_dcIn[j - 1][k - 1];
                    // SUMA
                    dJp_daIn[j - 1][k - 1] = dJp_daIn[j - 1][k - 1] + dJn_daIn[j - 1][k - 1];
                    dJp_dcIn[j - 1][k - 1] = dJp_dcIn[j - 1][k - 1] + dJn_dcIn[j - 1][k - 1];
                }
            }
        }

        // e) Epoch summary
        APE[g] = 0.0;
        for (int i = 2 + backwards; i < nData; i++) {
            APE[g] = APE[g] + sqrt(En[i] * En[i]) / sqrt(OUTPUT[i] * OUTPUT[i]);
        }
        APE[g] = APE[g] * 100 / (nData - long long(backwards) - 1);
        if (APE[g] <= APEmin) {
            APEmin = APE[g];
            for (int k = 0; k < nInputs; k++) {
                for (int j = 0; j < nFuzzy; j++) {
                    aInfinal[j][k] = aIne[j][k];
                    cInfinal[j][k] = cIne[j][k];
                }
            }
            for (int j = 0; j < nRules; j++) {
                aOutfinal[j] = aOute[j];
                cOutfinal[j] = cOute[j];
            }
            epochFlag = g;
        }

        // f) New step size
        if (g > 4) {
            if (APE[g] < APE[g - 1]) {
                if (APE[g - 1] < APE[g - 2]) {
                    if (APE[g - 2] < APE[g - 3]) {
                        if (APE[g - 3] < APE[g - 4]) {
                            K = K * (1.0 + growthRate);
                        }
                    }
                }
            }
            else {
                if (APE[g - 1] < APE[g - 2]) {
                    if (APE[g - 2] > APE[g - 3]) {
                        if (APE[g - 3] < APE[g - 4]) {
                            K = K * (1.0 - growthRate);
                        }
                    }
                }
            }
        }
        if (K > maxK) {
            K = maxK;
        }

        // g) New consequent parameters
        for (int j = 0; j < nRules; j++) {
            // Change in parameters
            aOute[j] = aOute[j] - K * sign(dJp_daOut[j]);
            cOute[j] = cOute[j] - K * sign(dJp_dcOut[j]);
            if (fabs(aOute[j]) < tol) {
                aOute[j] = tol;
            }
            if (fabs(cOute[j]) < tol) {
                cOute[j] = tol;
            }
        }

        // h) New premise parameters
        /*
        for (int k = 0; k < nInputs; k++) {
            for (int j = 0; j < nFuzzy; j++) {
                // Change in parameters
                aIne[j][k] = aIne[j][k] - K * sign(dJp_daIn[j][k]);
                cIne[j][k] = cIne[j][k] - K * sign(dJp_dcIn[j][k]);
                if (fabs(aIne[j][k]) < tol) {
                    aIne[j][k] = tol;
                }
                if (fabs(cIne[j][k]) < tol) {
                    cIne[j][k] = tol;
                }
            }
        }*/
        printf("\nEpoch: %04d.\t APE: %02.10f.\t K: %.10f.\t APEmin: %d,\t %.7f.\t %02.4f seconds.", g, APE[g], K, epochFlag, APEmin, (double(clock()) - double(epochTime)) / CLOCKS_PER_SEC);
    }

    //----------- Step 8: Plotting training results ---------//
    printf("\n\nStep 8: Plotting training results");

    FILE* anfisFile;
    anfisFile = fopen("ANFIStraining_Results.m", "w");
    if (anfisFile != NULL) {
        for (int k = 1; k <= nInputs; k++) {
            for (int j = 1; j <= nFuzzy; j++) {
                fprintf(anfisFile, "\naInfinal_cpp(%d,%d)\t= %03.15f;", j, k, aInfinal[j - 1][k - 1]);
            }
        }
        for (int k = 1; k <= nInputs; k++) {
            for (int j = 1; j <= nFuzzy; j++) {
                fprintf(anfisFile, "\ncInfinal_cpp(%d,%d)\t= %03.15f;", j, k, cInfinal[j - 1][k - 1]);
            }
        }
        for (int j = 1; j <= nRules; j++) {
            fprintf(anfisFile, "\naOutfinal_cpp(%d,1) \t= %03.15f;", j, aOutfinal[j - 1]);
        }
        for (int j = 1; j <= nRules; j++) {
            fprintf(anfisFile, "\ncOutfinal_cpp(%d,1) \t= %03.15f;", j, cOutfinal[j - 1]);
        }
        for (int j = 1; j <= maxEpochs; j++) {
            fprintf(anfisFile, "\nAPE_cpp(%d,1) \t= %03.15f;", j, APE[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(anfisFile, "\nO5_cpp(%d) \t= %03.15f;", j, O5[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(anfisFile, "\nOUTPUT_cpp(%d) \t= %03.15f;", j, OUTPUT[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(anfisFile, "\nINPUT1_cpp(%d) \t= %03.15f;", j, INPUT1[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(anfisFile, "\nINPUT2_cpp(%d) \t= %03.15f;", j, INPUT2[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(anfisFile, "\nINPUT3_cpp(%d) \t= %03.15f;", j, INPUT3[j - 1]);
        }
        for (int j = 1; j <= nData; j++) {
            fprintf(anfisFile, "\nINPUT4_cpp(%d) \t= %03.15f;", j, INPUT4[j - 1]);
        }
        fprintf(anfisFile, "\nAPEmin_cpp \t= %03.15f;", APEmin);

        printf("\n\t¡Success!");
    }
    fclose(anfisFile);


    //----------- Step 9: Plotting training results ---------//
    FILE* ANFIS_parameters;
    ANFIS_parameters = fopen("ANFIS_parameters.txt", "w");
    if (ANFIS_parameters != NULL) {
        for (int k = 0; k < nInputs; k++) {
            for (int j = 0; j < nFuzzy; j++) {
                fprintf(ANFIS_parameters, "\naIne[%d][%d]\t= %33.30f;", j, k, aInfinal[j][k]);
            }
        }
        for (int k = 0; k < nInputs; k++) {
            for (int j = 0; j < nFuzzy; j++) {
                fprintf(ANFIS_parameters, "\ncIne[%d][%d]\t= %33.30f;", j, k, cInfinal[j][k]);
            }
        }
        for (int j = 0; j < nRules; j++) {
            fprintf(ANFIS_parameters, "\naOute[%d] \t= %33.30f;", j, aOutfinal[j]);
        }
        for (int j = 0; j < nRules; j++) {
            fprintf(ANFIS_parameters, "\ncOute[%d] \t= %33.30f;", j, cOutfinal[j]);
        }
        printf("\n\t¡Success!");
    }
    fclose(ANFIS_parameters);
    //----------- Step 10: Finish ---------//
    printf("\n\nStep 10: Finish");
    printf("\n\n\n");



    return 0;
}















//-------------------------------| 
//        CUDA WRAPPERS          |
//-------------------------------| 





//-------------------------------| 
//          FUNCTIONS            |
//-------------------------------| 
void mapping(double minIn, double maxIn, double minOut, double maxOut) {

}

double h(double t) {
    if (t >= 0.0) {
        return 1.0;
    }
    else {
        return 0.0;
    }
}

double r(double t) {
    return h(t) * t;
}

void fill(double* arr, int n, int m, double val) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            *(arr + (long long(i) * long long(m) + long long(j))) = val;
        }
    }
}

double sign(double x) {
    if (x >= 0.0) {
        return 1.0;
    }
    else {
        return -1.0;
    }
}

double map(double input, double minIn, double maxIn, double minOut, double maxOut) {
    return ((input - minIn) / (maxIn - minIn)) * (maxOut - minOut) + minOut;
}

double u1(double t, double amp1) {
    return h(t) * -amp1 + r(t - 10) * amp1 * 0.0075 - r(t - 280) * amp1 * 0.0075 - (+r(t - 290) * amp1 * 0.0075 - r(t - 560) * amp1 * 0.0075);

}

double u2(double t, double amp2) {
    return h(t) * 0.5 + r(t - 10) * amp2 * 0.015 - r(t - 100) * amp2 * 0.03 + r(t - 280) * amp2 * 0.03 - r(t - 450) * amp2 * 0.03;
}

double u3(double t, double amp3) {
    return h(t) * -amp3 + r(t - 10) * amp3 * 0.0075 - r(t - 280) * amp3 * 0.0075 - h(t - 290) * amp3 * 2 + r(t - 290) * amp3 * 0.0075 - r(t - 560) * amp3 * 0.0075;
}

double gauss(double x, double a, double c) {
    return exp(-(x - c) * (x - c) / (a * a));
}

double sigmoid(double x, double a, double c) {
    return 1.0 / (1.0 + exp(-(x - c) / a));
}

double invSigmoid(double x, double a, double c) {
    return  c - a * log(1.0 / x - 1.0);
}

double dGauss_da(double x, double a, double c) {
    return (2.0 * exp(-(-c + x) * (-c + x) / (a * a)) * (-c + x) * (-c + x)) / (a * a * a);
}

double dGauss_dc(double x, double a, double c) {
    return (2.0 * exp(-(-c + x) * (-c + x) / (a * a)) * (-c + x)) / (a * a);
}

double dinvSigmoid_da(double x, double a, double c) {
    return -log(1.0 / x - 1.0);
}

double dinvSigmoid_dc(double x, double a, double c) {
    return 1.0;
}




// END





