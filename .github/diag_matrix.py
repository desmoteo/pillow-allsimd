"""Temporary: why the colour matrix differs between the scalar and AVX2
tiers of the -C simd=avx2 build on this runner."""
import glob, os, struct, subprocess, sys, sysconfig

sys.path.insert(0, os.getcwd())  # the checkout's root, for the Tests package

from PIL import Image  # noqa: E402
from Tests.simd_shapes import SHAPES  # noqa: E402
from Tests.test_simd_tiers import _inputs  # noqa: E402


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True)


print(sys.version)
print("CFLAGS:", sysconfig.get_config_var("CFLAGS"))
print("tiers:", Image.core.simd_tiers, "active:", Image.core.simd_kernels)
sh("gcc --version | head -1; which gcc; readlink -f $(which gcc)")
sh("grep -m1 'model name' /proc/cpuinfo; grep -m1 -o -E 'avx512[a-z]*|fma' /proc/cpuinfo | sort -u | tr '\\n' ' '; echo")
sh("gcc -Q --help=target -O3 -msse4.1 -mavx2 | grep -E 'march=|mtune=|mfma |mfpmath=|mprefer-vector-width'")
sh("gcc -Q --help=optimizers -O3 -msse4.1 -mavx2 | grep -E 'ffp-contract|ftrapping-math|fassociative-math|funsafe-math|fexcess-precision'")
sh("gcc -dumpspecs | grep -c -E 'march|mtune|ffp'; env | grep -E '^(CFLAGS|CC|CPPFLAGS|LDFLAGS)=' || echo 'no CFLAGS/CC in the environment'")
so = glob.glob(os.path.join(os.path.dirname(Image.__file__), "_imaging.*.so"))[0]
print("module:", so)
sh(f"objdump -d --no-show-raw-insn {so} | grep -c -E 'vfn?m(add|sub)' ; echo '^ FMA instructions in the module'")
for fn in ("convert_matrix_rgb_row_simd", "ImagingConvertMatrix"):
    sh(f"objdump -d --no-show-raw-insn {so} | awk '/<{fn}>:/,/^$/' > /tmp/{fn}.s; wc -l < /tmp/{fn}.s; grep -c -E 'vfn?m(add|sub)' /tmp/{fn}.s; grep -o -E '\\b(v?(mul|add|cvtt?)[a-z0-9]+)\\b' /tmp/{fn}.s | sort | uniq -c | sort -rn | head -12")
sh("grep -E -- ' -c src/libImaging/(Matrix|SIMDKernels).c ' install.log | sed -E 's/ -I[^ ]+//g' | cut -c1-400")

M = (1.9, -0.6, -0.4, 3.5, -0.3, 1.7, -0.5, -12, 0.2, -1.1, 2.2, 0)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def clip(v):
    return 0 if v <= 0 else 255 if v >= 255 else int(v)


def candidates(m, r, g, b):
    m = [f32(x) for x in m]
    plain = f32(f32(f32(f32(m[0] * r) + f32(m[1] * g)) + f32(m[2] * b)) + m[3])
    out = {"separate multiplies and adds": clip(f32(plain + 0.5))}
    # products fused into the additions, in the orders a compiler may choose
    a = f32(m[0] * r + f32(m[1] * g)); a = f32(m[2] * b + a); a = f32(a + m[3])
    out["fused, first product into the second"] = clip(f32(a + 0.5))
    a = f32(f32(m[0] * r) + m[1] * g); a = f32(a + m[2] * b); a = f32(a + m[3])
    out["fused, left to right"] = clip(f32(a + 0.5))
    exact = m[0] * r + m[1] * g + m[2] * b + m[3] + 0.5
    out["exact"] = clip(exact)
    return out, exact


for shape in ("landscape", "portrait-tall"):
    outs = {}
    for tier in Image.core.simd_tiers:
        Image.core.set_simd(tier)
        rgb = _inputs(SHAPES[shape])["RGB"]
        outs[tier] = (rgb.tobytes(), rgb.convert("RGB", M).tobytes())
    (in_s, s), (in_a, a) = outs["scalar"], outs["avx2"]
    w, h = SHAPES[shape]
    diffs = [i for i in range(len(s)) if i % 4 != 3 and s[i] != a[i]]
    print(f"{shape} {w}x{h}: inputs equal {in_s == in_a}; {len(diffs)} of {len(s) * 3 // 4} values differ; by x mod 8: "
          f"{sorted({(i // 4) % w % 8 for i in diffs})}; largest difference {max((abs(s[i] - a[i]) for i in diffs), default=0)}")
    for i in diffs[:12]:
        px, band = i // 4, i % 4
        r, g, b = in_s[px * 4], in_s[px * 4 + 1], in_s[px * 4 + 2]
        c, exact = candidates(M[band * 4:band * 4 + 4], r, g, b)
        print(f"  x {px % w:3} y {px // w:3} band {band} rgb {(r, g, b)}: scalar {s[i]} avx2 {a[i]} exact {exact:.6f} {c}")
