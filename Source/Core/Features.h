#pragma once
// The detector's input picture of a sound (port of the prototype's feat.py):
// 4 layers x 256 log-spaced bands (30 Hz - 20 kHz) x 20 ms frames:
//   0 log mid level, 1 side relative to mid, 2 left/right coherence, 3 mid minus its time median.

#include "Dsp.h"

namespace dr
{

struct FeatureMap
{
    static constexpr int NB = 256;
    int frames = 0;
    std::vector<float> v;     // [layer][band][frame]
    float& at (int c, int b, int t) { return v[((size_t) c * NB + (size_t) b) * (size_t) frames + (size_t) t]; }
    float at (int c, int b, int t) const { return v[((size_t) c * NB + (size_t) b) * (size_t) frames + (size_t) t]; }
};

namespace feat
{
constexpr int NF = 1024, HOP = 480, NB = FeatureMap::NB;

// Band-averaging weights over the rfft bins (each band = mean of its bins, or the nearest bin).
struct Bands
{
    std::vector<std::vector<std::pair<int, double>>> w; // per band: (bin, weight)
    Bands()
    {
        const int nbins = NF / 2 + 1; w.resize (NB);
        for (int b = 0; b < NB; ++b)
        {
            const double lo = 30.0 * std::pow (20000.0 / 30.0, (double) b / NB), hi = 30.0 * std::pow (20000.0 / 30.0, (double) (b + 1) / NB);
            std::vector<int> sel;
            for (int k = 0; k < nbins; ++k) { const double f = (double) k * SR / NF; if (f >= lo && f < hi) sel.push_back (k); }
            if (sel.empty())
            {
                const double c = std::sqrt (lo * hi); int best = 0; double bd = 1e300;
                for (int k = 0; k < nbins; ++k) { const double d = std::abs ((double) k * SR / NF - c); if (d < bd) { bd = d; best = k; } }
                sel.push_back (best);
            }
            for (int k : sel) w[(size_t) b].push_back ({ k, 1.0 / (double) sel.size() });
        }
    }
};
inline const Bands& bands() { static Bands b; return b; }

inline float toHalf (float x)   // round to float16 precision (the models were trained on float16 features)
{
    if (! std::isfinite (x) || x == 0.0f) return x;
    int e; const float m = std::frexp (x, &e);            // x = m * 2^e, 0.5 <= |m| < 1
    if (e < -13) { const float q = std::ldexp (1.0f, -24); return std::round (x / q) * q; } // subnormal
    (void) m;
    const float q = std::ldexp (1.0f, e - 11);       // 11 significant bits
    return std::nearbyint (x / q) * q;                // round half to even, like numpy
}
} // namespace feat

inline FeatureMap features (const Stereo& y)
{
    using namespace feat;
    const auto& B = bands();
    const long n = (long) y.size();
    const int T0 = n >= NF ? (int) (1 + (n - NF) / HOP) : 0;
    FeatureMap F; F.frames = T0 / 2; F.v.assign ((size_t) 4 * NB * (size_t) std::max (0, F.frames), 0.0f);
    if (T0 < 2) return F;
    const Vec win = hannPeriodic (NF);
    double wsum = 0; for (double a : win) wsum += a;
    const int nbins = NF / 2 + 1;
    std::vector<std::vector<double>> pM ((size_t) NB, Vec ((size_t) T0)), pS = pM, cr = pM, pl = pM, pr = pM;
    Vec re (NF), im (NF), re2 (NF), im2 (NF);
    std::vector<std::complex<double>> ZL ((size_t) nbins), ZR ((size_t) nbins);
    for (int t = 0; t < T0; ++t)
    {
        const long s = (long) t * HOP;
        for (int i = 0; i < NF; ++i) { re[(size_t) i] = y.l[(size_t) (s + i)] * win[(size_t) i]; im[(size_t) i] = 0; re2[(size_t) i] = y.r[(size_t) (s + i)] * win[(size_t) i]; im2[(size_t) i] = 0; }
        fft (re, im); fft (re2, im2);
        for (int k = 0; k < nbins; ++k) { ZL[(size_t) k] = { re[(size_t) k] / wsum, im[(size_t) k] / wsum }; ZR[(size_t) k] = { re2[(size_t) k] / wsum, im2[(size_t) k] / wsum }; }
        for (int b = 0; b < NB; ++b)
        {
            double m = 0, sd = 0, c = 0, a = 0, d = 0;
            for (auto [k, w] : B.w[(size_t) b])
            {
                const auto L = ZL[(size_t) k], R = ZR[(size_t) k];
                const auto M = 0.5 * (L + R), S = 0.5 * (L - R);
                m += w * std::norm (M); sd += w * std::norm (S);
                c += w * (L * std::conj (R)).real(); a += w * std::norm (L); d += w * std::norm (R);
            }
            pM[(size_t) b][(size_t) t] = m; pS[(size_t) b][(size_t) t] = sd; cr[(size_t) b][(size_t) t] = c; pl[(size_t) b][(size_t) t] = a; pr[(size_t) b][(size_t) t] = d;
        }
    }
    auto conv3 = [&] (const Vec& v) { Vec o (v.size()); for (size_t i = 0; i < v.size(); ++i) { double s = v[i]; if (i > 0) s += v[i - 1]; if (i + 1 < v.size()) s += v[i + 1]; o[i] = s / 3.0; } return o; };
    std::vector<Vec> lm ((size_t) NB), ls ((size_t) NB), coh ((size_t) NB);
    double top = -1e300;
    for (int b = 0; b < NB; ++b)
    {
        lm[(size_t) b].resize ((size_t) T0);
        for (int t = 0; t < T0; ++t) { lm[(size_t) b][(size_t) t] = 10.0 * std::log10 (pM[(size_t) b][(size_t) t] + 1e-10); top = std::max (top, lm[(size_t) b][(size_t) t]); }
        const Vec c3 = conv3 (cr[(size_t) b]), l3 = conv3 (pl[(size_t) b]), r3 = conv3 (pr[(size_t) b]);
        coh[(size_t) b].resize ((size_t) T0);
        for (int t = 0; t < T0; ++t) coh[(size_t) b][(size_t) t] = c3[(size_t) t] / std::sqrt (l3[(size_t) t] * r3[(size_t) t] + 1e-12);
    }
    std::vector<Vec> fine ((size_t) NB);
    for (int b = 0; b < NB; ++b)
    {
        auto& L = lm[(size_t) b];
        for (auto& v : L) v = std::max (v, top - 80.0);
        ls[(size_t) b].resize ((size_t) T0);
        for (int t = 0; t < T0; ++t) ls[(size_t) b][(size_t) t] = std::clamp (10.0 * std::log10 (pS[(size_t) b][(size_t) t] + 1e-10) - L[(size_t) t], -40.0, 10.0);
        const double med = median (L);
        fine[(size_t) b].resize ((size_t) T0);
        for (int t = 0; t < T0; ++t) fine[(size_t) b][(size_t) t] = L[(size_t) t] - med;
    }
    for (int b = 0; b < NB; ++b)
        for (int t = 0; t < F.frames; ++t)
        {
            const size_t a = (size_t) (2 * t), c = a + 1;
            // frame pairs averaged in double, then rounded once to float16, as in the prototype
            auto pair = [&] (const Vec& v, double sc, double off) { return feat::toHalf ((float) (0.5 * ((v[a] * sc + off) + (v[c] * sc + off)))); };
            F.at (0, b, t) = pair (lm[(size_t) b], 1.0 / 20.0, 2.0 - top / 20.0);
            F.at (1, b, t) = pair (ls[(size_t) b], 0.1, 0.0);
            F.at (2, b, t) = pair (coh[(size_t) b], 1.0, 0.0);
            F.at (3, b, t) = pair (fine[(size_t) b], 0.1, 0.0);
        }
    return F;
}

} // namespace dr
