/*
  Vectorial
  Copyright (c) 2010 Mikko Lehtonen
  Licensed under the terms of the two-clause BSD License (see LICENSE)
*/
#ifndef VECTORIAL_SIMD4X4F_NEON_H
#define VECTORIAL_SIMD4X4F_NEON_H

vectorial_inline void simd4x4f_transpose_inplace(simd4x4f* s) {
    float32x4x2_t zip_13 = vzipq_f32(s->x, s->z);
    float32x4x2_t zip_24 = vzipq_f32(s->y, s->w);

    float32x4x2_t rows_xy = vzipq_f32(zip_13.val[0], zip_24.val[0]);
    float32x4x2_t rows_zw = vzipq_f32(zip_13.val[1], zip_24.val[1]);

    s->x = rows_xy.val[0];
    s->y = rows_xy.val[1];
    s->z = rows_zw.val[0];
    s->w = rows_zw.val[1];
}

vectorial_inline void simd4x4f_transpose(const simd4x4f *s, simd4x4f *out) {
    *out=*s;
    simd4x4f_transpose_inplace(out);
}



#endif
