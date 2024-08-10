#ifndef __FC_PARAMS__
#define __FC_PARAMS__

#define FC_IN_DIM (32)
#define FC_OUT_DIM (6)

// clang-format off

const float FC_W[6][32] = {
   {2.367241e-01, 8.488032e-01, -7.140010e-02, -7.443851e-01, 1.387011e-01, 5.829725e-01, 9.719965e-02, 4.349915e-01,
    -8.660481e-02, 1.464107e-01, 7.366611e-02, 3.431566e-03, -6.683898e-01, 5.643771e-01, -2.897255e-01, -5.134247e-02,
    3.666034e-01, 1.034972e-01, 6.874534e-01, -7.792107e-03, -3.656393e-01, 1.044935e-01, -2.165809e-01, -4.206351e-01,
    -6.057175e-02, 8.311038e-02, 3.282499e-01, 1.065499e-01, 3.997944e-01, 2.021558e-01, 2.215551e-01, 2.232014e-01},
   {-1.340621e+00, -1.428982e+00, -3.776577e-01, -7.275188e-01, -5.593478e-02, 8.514512e-01, 1.045755e-02, -1.310953e+00,
    -9.044311e-02, 3.671524e-01, -2.921892e-01, 3.431801e-01, -8.190925e-01, -1.684011e+00, 1.097420e+00, 1.063884e-01,
    -1.816654e+00, -5.557809e-01, -1.422901e+00, -8.243779e-02, -7.771053e-01, -9.515904e-02, -7.850571e-01, -1.753932e-01,
    1.864210e-01, -1.109782e-01, 2.656964e-01, 2.902703e-01, -1.373768e+00, -1.517846e+00, 3.100403e-01, 3.819744e-01},
   {-1.585818e+00, -1.616139e+00, -5.481097e-01, -9.473088e-01, -3.336906e-01, 6.130480e-01, 1.254109e-01, -1.829803e+00,
    -5.580369e-02, 3.855740e-01, -3.402322e-01, 2.238987e-01, -6.319513e-01, -2.008326e+00, -1.136765e+00, 1.598088e-01,
    1.190976e+00, -5.470506e-02, -1.830897e-01, -1.861736e-01, -7.772707e-01, -2.451072e-01, 1.739596e+00, 6.087238e-01,
    9.615327e-02, -2.370944e-01, 1.091310e+00, 2.470271e-01, -1.326948e+00, 7.345777e-01, 3.504354e-01, -1.905532e+00},
   {4.291048e-01, -1.880272e+00, -3.068003e-01, 1.160250e+00, -6.100370e-01, -1.728421e+00, -2.120382e-01, 6.223533e-01,
    -4.230410e-01, -1.033418e+00, -3.914648e-01, 1.359578e-01, -6.371011e-01, 4.444395e-01, -4.389924e-01, -7.332598e-02,
    -1.446761e+00, -4.424990e-01, -1.719493e+00, -1.903224e-02, 1.543655e+00, 3.257615e-02, 1.116120e+00, 2.812027e-01,
    1.335952e-01, 1.197612e-01, -8.311491e-02, 8.701118e-02, 6.390706e-01, -2.512877e-02, 3.320628e-01, 1.671530e-01},
   {5.420404e-01, 1.198645e+00, 2.064476e-01, 1.786274e+00, 3.670774e-01, -7.988220e-01, -2.509862e-01, 5.799395e-01,
    5.368005e-02, 4.447356e-01, 1.222515e-01, -8.876305e-01, 2.101363e+00, 5.917464e-01, -5.104939e-01, -2.927724e-01,
    1.082966e+00, 7.924844e-01, 5.542532e-01, -8.282033e-02, -7.754298e-01, -5.355914e-02, -1.368673e+00, 9.678677e-01,
    4.142547e-01, 1.234834e-01, -2.146572e+00, 3.085512e-01, 6.758908e-01, 1.284060e-01, -1.839678e+00, 4.644545e-02},
   {1.919786e-01, 1.007310e+00, 9.611736e-02, 1.041720e+00, -7.770877e-01, -1.860442e+00, -4.012137e-01, 3.535838e-01,
    3.977487e-01, -1.685615e+00, -2.049714e-01, 1.341641e-01, 1.334518e+00, 8.073455e-01, 1.769507e+00, -3.496643e-01,
    -1.740301e+00, -5.938721e-01, 2.285216e-01, -8.417810e-02, 1.704039e+00, 2.266375e-02, -9.683805e-01, -1.479149e+00,
    -1.740088e+00, -2.922066e-01, -4.688823e-01, -1.779415e+00, 4.388652e-01, -2.492617e-01, 4.465487e-02, 4.069645e-01},
};

const float FC_B[6] = 
   {1.724649e+00, -6.816645e-01, -1.745888e+00, -1.156646e+00, -7.215285e-01, -6.078057e-01};

// clang-format on


#endif // __FC_PARAMS__

