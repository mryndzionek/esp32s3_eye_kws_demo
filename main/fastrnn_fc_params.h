#ifndef __FASTRNN_FC_PARAMS__
#define __FASTRNN_FC_PARAMS__

#define FC_IN_DIM (32)
#define FC_OUT_DIM (6)

// clang-format off

const float FC_W[6][32] = {
   { 2.36724108e-01,  8.48803222e-01, -7.14000985e-02, -7.44385064e-01,  1.38701141e-01,  5.82972467e-01,  9.71996486e-02,  4.34991539e-01,
    -8.66048113e-02,  1.46410733e-01,  7.36661106e-02,  3.43156606e-03, -6.68389797e-01,  5.64377069e-01, -2.89725512e-01, -5.13424687e-02,
     3.66603374e-01,  1.03497177e-01,  6.87453449e-01, -7.79210683e-03, -3.65639329e-01,  1.04493484e-01, -2.16580868e-01, -4.20635074e-01,
    -6.05717525e-02,  8.31103772e-02,  3.28249902e-01,  1.06549934e-01,  3.99794400e-01,  2.02155843e-01,  2.21555099e-01,  2.23201409e-01},
   {-1.34062052e+00, -1.42898190e+00, -3.77657741e-01, -7.27518797e-01, -5.59347831e-02,  8.51451218e-01,  1.04575502e-02, -1.31095266e+00,
    -9.04431120e-02,  3.67152393e-01, -2.92189151e-01,  3.43180150e-01, -8.19092512e-01, -1.68401134e+00,  1.09742010e+00,  1.06388398e-01,
    -1.81665397e+00, -5.55780888e-01, -1.42290056e+00, -8.24377909e-02, -7.77105331e-01, -9.51590389e-02, -7.85057127e-01, -1.75393194e-01,
     1.86420992e-01, -1.10978179e-01,  2.65696406e-01,  2.90270329e-01, -1.37376821e+00, -1.51784611e+00,  3.10040325e-01,  3.81974429e-01},
   {-1.58581793e+00, -1.61613905e+00, -5.48109710e-01, -9.47308838e-01, -3.33690643e-01,  6.13048017e-01,  1.25410900e-01, -1.82980263e+00,
    -5.58036938e-02,  3.85574013e-01, -3.40232223e-01,  2.23898709e-01, -6.31951332e-01, -2.00832558e+00, -1.13676548e+00,  1.59808829e-01,
     1.19097590e+00, -5.47050610e-02, -1.83089703e-01, -1.86173558e-01, -7.77270734e-01, -2.45107174e-01,  1.73959601e+00,  6.08723760e-01,
     9.61532742e-02, -2.37094447e-01,  1.09130991e+00,  2.47027144e-01, -1.32694793e+00,  7.34577656e-01,  3.50435376e-01, -1.90553164e+00},
   { 4.29104775e-01, -1.88027215e+00, -3.06800276e-01,  1.16024971e+00, -6.10036969e-01, -1.72842062e+00, -2.12038219e-01,  6.22353315e-01,
    -4.23041016e-01, -1.03341794e+00, -3.91464829e-01,  1.35957778e-01, -6.37101114e-01,  4.44439471e-01, -4.38992441e-01, -7.33259767e-02,
    -1.44676054e+00, -4.42498952e-01, -1.71949267e+00, -1.90322399e-02,  1.54365516e+00,  3.25761512e-02,  1.11611998e+00,  2.81202674e-01,
     1.33595228e-01,  1.19761229e-01, -8.31149071e-02,  8.70111808e-02,  6.39070570e-01, -2.51287688e-02,  3.32062781e-01,  1.67153001e-01},
   { 5.42040408e-01,  1.19864547e+00,  2.06447557e-01,  1.78627372e+00,  3.67077380e-01, -7.98822045e-01, -2.50986159e-01,  5.79939485e-01,
     5.36800511e-02,  4.44735587e-01,  1.22251466e-01, -8.87630522e-01,  2.10136342e+00,  5.91746390e-01, -5.10493875e-01, -2.92772353e-01,
     1.08296645e+00,  7.92484403e-01,  5.54253221e-01, -8.28203261e-02, -7.75429845e-01, -5.35591356e-02, -1.36867344e+00,  9.67867732e-01,
     4.14254665e-01,  1.23483442e-01, -2.14657164e+00,  3.08551192e-01,  6.75890803e-01,  1.28406033e-01, -1.83967757e+00,  4.64454480e-02},
   { 1.91978619e-01,  1.00731003e+00,  9.61173624e-02,  1.04172039e+00, -7.77087748e-01, -1.86044180e+00, -4.01213676e-01,  3.53583843e-01,
     3.97748679e-01, -1.68561530e+00, -2.04971358e-01,  1.34164065e-01,  1.33451772e+00,  8.07345510e-01,  1.76950657e+00, -3.49664301e-01,
    -1.74030054e+00, -5.93872130e-01,  2.28521585e-01, -8.41781050e-02,  1.70403886e+00,  2.26637479e-02, -9.68380511e-01, -1.47914898e+00,
    -1.74008834e+00, -2.92206585e-01, -4.68882293e-01, -1.77941513e+00,  4.38865215e-01, -2.49261737e-01,  4.46548723e-02,  4.06964481e-01},
};

const float FC_B[6] = 
   { 1.72464907e+00, -6.81664467e-01, -1.74588788e+00, -1.15664625e+00, -7.21528471e-01, -6.07805729e-01};

// clang-format on


#endif // __FASTRNN_FC_PARAMS__
