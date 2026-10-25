/* NEON 3x3 and 5x5 float kernels for 8-bit images, included from Filter.c.
   Used when the coefficients don't fit the 16-bit integer kernels.

   Same arithmetic as the generic float code in Filter.c as GCC compiles it
   for arm64, which fuses each kernel row into multiply-adds:
       row = fma(p[size - 1], k[size - 1], ... fma(p[1], k[1], p[0] * k[0]))
       ss = offset + 0.5, then ss += row for each kernel row in order
   and then clamps to 0..255 and truncates. The bytes left over at the end of
   a row use fmaf() for the same rounding.

   Kernel row 0 applies to image row y + R, the last kernel row to y - R,
   and the R border rows and columns are copied, as in the generic code.
   bpp is 1 or 4; for 2- and 3-band images the unused bytes are 0. */

static inline float32x4_t
filter_f_lanes(uint8x16_t v, int i) {
    uint16x8_t h = i < 2 ? vmovl_u8(vget_low_u8(v)) : vmovl_high_u8(v);
    uint32x4_t w = (i & 1) ? vmovl_high_u16(h) : vmovl_u16(vget_low_u16(h));
    return vcvtq_f32_u32(w);
}

static inline uint8_t
filter_f_clip(float ss) {
    ss = ss < 0.0f ? 0.0f : ss;
    ss = ss > 255.0f ? 255.0f : ss;
    return (uint8_t)ss;
}

static void
ImagingFilterNxNf_neon(
    Imaging imOut, Imaging im, const float *kernel, float offset, int size, int bpp
) {
    int y, r, d, i, b;
    int R = size / 2;
    int xsize = im->xsize, ysize = im->ysize;
    int bytes = xsize * bpp;
    const UINT8 *rows[5];
    uint8x16_t mask = vdupq_n_u8(0xff);
    UINT8 mask_bytes[4] = {0xff, 0xff, 0xff, 0xff};

    if (bpp == 4 && im->bands == 2) {
        mask_bytes[1] = mask_bytes[2] = 0;
    } else if (bpp == 4 && im->bands == 3) {
        mask_bytes[3] = 0;
    }
    if (bpp == 4) {
        UINT32 m;
        memcpy(&m, mask_bytes, sizeof(m));
        mask = vreinterpretq_u8_u32(vdupq_n_u32(m));
    }
    offset += 0.5f;

    for (y = 0; y < R; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
    for (y = R; y < ysize - R; y++) {
        UINT8 *out = (UINT8 *)imOut->image[y];
        const UINT8 *in0 = (UINT8 *)im->image[y];
        for (r = 0; r < size; r++) {
            rows[r] = (UINT8 *)im->image[y + R - r];
        }

        memcpy(out, in0, R * bpp);
        b = R * bpp;
        for (; b < bytes - R * bpp - 15; b += 16) {
            float32x4_t ss[4];
            for (i = 0; i < 4; i++) {
                ss[i] = vdupq_n_f32(offset);
            }
            for (r = 0; r < size; r++) {
                const float *k = &kernel[r * size];
                float32x4_t row[4];
                uint8x16_t p = vld1q_u8(rows[r] + b - R * bpp);
                for (i = 0; i < 4; i++) {
                    row[i] = vmulq_n_f32(filter_f_lanes(p, i), k[0]);
                }
                for (d = 1; d < size; d++) {
                    p = vld1q_u8(rows[r] + b + (d - R) * bpp);
                    for (i = 0; i < 4; i++) {
                        row[i] = vfmaq_n_f32(row[i], filter_f_lanes(p, i), k[d]);
                    }
                }
                for (i = 0; i < 4; i++) {
                    ss[i] = vaddq_f32(ss[i], row[i]);
                }
            }
            uint16x8_t lo = vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(
                    vminq_f32(vmaxq_f32(ss[0], vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f))
                )),
                vmovn_u32(vcvtq_u32_f32(
                    vminq_f32(vmaxq_f32(ss[1], vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f))
                ))
            );
            uint16x8_t hi = vcombine_u16(
                vmovn_u32(vcvtq_u32_f32(
                    vminq_f32(vmaxq_f32(ss[2], vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f))
                )),
                vmovn_u32(vcvtq_u32_f32(
                    vminq_f32(vmaxq_f32(ss[3], vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f))
                ))
            );
            vst1q_u8(
                &out[b], vandq_u8(vcombine_u8(vmovn_u16(lo), vmovn_u16(hi)), mask)
            );
        }
        for (; b < bytes - R * bpp; b++) {
            float ss = offset;
            for (r = 0; r < size; r++) {
                const float *k = &kernel[r * size];
                const UINT8 *p = rows[r] + b - R * bpp;
                float row = (float)p[0] * k[0];
                for (d = 1; d < size; d++) {
                    row = fmaf((float)p[d * bpp], k[d], row);
                }
                ss += row;
            }
            out[b] = mask_bytes[b % bpp] ? filter_f_clip(ss) : 0;
        }
        memcpy(out + b, in0 + b, R * bpp);
    }
    for (; y < ysize; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
}
