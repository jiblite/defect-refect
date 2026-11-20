#pragma once
// The curve reader: what the FILTER engine (Filter.h) looks at, and the small net that reads a curve over the bands
// from it. (A reader of the same kind was taught an exciter's lift and withdrawn: Research/exciter.)
//   energies()   a sound's band energies frame by frame: 104 bands a twelfth of an octave apart, 50 Hz .. 19 kHz,
//                both sides together (Hann frames of 8192 every 2048, none hanging over either end)
//   curves()     six curves over the bands from them: per band its mean level and its level in the quietest frame,
//                at the 10th, 50th and 90th percentile of the frames, and in the loudest frame, in dB against the
//                sound's own level; frames more than 50 dB under the loudest do not count
//   Net          five 1-D convolutions along the bands (5 wide, ever further apart) and two heads: a curve in dB band
//                by band, and one number that says how sure the reader is that the thing it was taught is there
//   Reader       several such nets read together
// A fixed filter moves all six curves of a band by the same amount, whatever the sound does: what the reader was
// taught is to tell that from what sounds look like by themselves (Research/filter). Nothing here depends on how loud
// the sound is.
#include "Tone.h"
#include <cstring>
#include <stdexcept>

namespace dr
{
namespace reader
{
using Stop = room::Stop;
constexpr int N = 8192, HOP = 2048, NBIN = N / 2 + 1, NBAND = 104, CURVES = 6;
constexpr double F0 = 50.0, FLOOR = -110.0, SPAN = 50.0;
constexpr int MAXFRAMES = 280;                 // 12 s; a longer sound is looked at in every second frame, every third ...
inline double bandHz (int b) { return F0 * std::pow (2.0, (double) b / 12.0); }

// Each band takes the bins around its centre with weights that fall off in a triangle: a twelfth of an octave to
// either side, and never less than a bin and a half.
struct Bands { std::vector<int> first; std::vector<std::vector<double>> w; };
inline const Bands& bands()
{
    static const Bands B = []
    {
        Bands b; b.first.resize ((size_t) NBAND); b.w.resize ((size_t) NBAND); const double binHz = SR / N;
        for (int i = 0; i < NBAND; ++i)
        {
            const double fc = bandHz (i), half = std::max (fc * (std::pow (2.0, 1.0 / 12.0) - 1.0), 1.5 * binHz);
            int lo = NBIN, hi = -1;
            for (int k = 0; k < NBIN; ++k) if (1.0 - std::abs ((double) k * binHz - fc) / half > 0.0) { lo = std::min (lo, k); hi = std::max (hi, k); }
            b.first[(size_t) i] = lo;
            for (int k = lo; k <= hi; ++k) b.w[(size_t) i].push_back (std::max (0.0, 1.0 - std::abs ((double) k * binHz - fc) / half));
        }
        return b;
    }();
    return B;
}

// Band energies per frame: [frame][NBAND], frames one after the other. Empty when the sound is shorter than one
// frame, or when given up.
struct Energies { int T = 0; std::vector<double> e; const double* row (int t) const { return e.data() + (size_t) t * (size_t) NBAND; } };
inline Energies energies (const Stereo& x, const Stop& stop = {})
{
    Energies E; const long n = (long) x.size(); if (n < N) return E;
    const long T = (n - N) / HOP + 1; static const Vec win = hannPeriodic ((size_t) N); const Bands& B = bands();
    const room::Transform tr (N); Vec re ((size_t) N), im ((size_t) N), P ((size_t) NBIN); E.e.assign ((size_t) T * (size_t) NBAND, 0.0);
    const double scale = 1.0 / ((double) (N / 2) * (double) (N / 2));      // as scipy.signal.stft scales its frames (by the window's sum)
    for (long t = 0; t < T; ++t)
    {
        if (stop && (t & 15) == 0 && stop()) return {};
        const long a = t * HOP;
        for (long i = 0; i < N; ++i) { const double w = win[(size_t) i]; re[(size_t) i] = x.l[(size_t) (a + i)] * w; im[(size_t) i] = x.r[(size_t) (a + i)] * w; }
        tr.run (re, im);        // left as the real part and right as the imaginary part: one transform for both
        for (int k = 0; k < NBIN; ++k)
        {
            const size_t u = (size_t) k, m = (size_t) ((N - k) & (N - 1));
            const double lr = 0.5 * (re[u] + re[m]), li = 0.5 * (im[u] - im[m]), rr = 0.5 * (im[u] + im[m]), ri = 0.5 * (re[m] - re[u]);
            P[u] = (lr * lr + li * li + rr * rr + ri * ri) * scale;
        }
        double* row = E.e.data() + (size_t) t * (size_t) NBAND;
        for (int b = 0; b < NBAND; ++b) { const auto& w = B.w[(size_t) b]; const int k0 = B.first[(size_t) b]; double s = 0; for (size_t j = 0; j < w.size(); ++j) s += w[j] * P[(size_t) k0 + j]; row[b] = s; }
    }
    E.T = (int) T; return E;
}

// The six curves: [CURVES][NBAND] in dB against the sound's level. gain (dB per band, or empty): the energies are
// looked at as if every band were turned by that much first (the reader's second look, with its first answer undone).
struct Curves { std::vector<float> c; int frames = 0; float at (int curve, int band) const { return c[(size_t) curve * (size_t) NBAND + (size_t) band]; } };
inline Curves curves (const Energies& E, const Vec& gain = {})
{
    Curves C; if (E.T < 1) return C;
    Vec g ((size_t) NBAND, 1.0); if (gain.size() == (size_t) NBAND) for (int b = 0; b < NBAND; ++b) g[(size_t) b] = std::pow (10.0, gain[(size_t) b] / 10.0);
    const int step = (E.T + MAXFRAMES - 1) / MAXFRAMES;        // a long sound: every step-th frame, so that the curves are of about as many frames as the reader was taught with
    Vec tot ((size_t) E.T, 0.0); double top = 0;
    for (int t = 0; t < E.T; t += step) { const double* r = E.row (t); double s = 0; for (int b = 0; b < NBAND; ++b) s += r[b] * g[(size_t) b]; tot[(size_t) t] = s; top = std::max (top, s); }
    std::vector<int> keep; { const double thr = top * std::pow (10.0, -SPAN / 10.0); for (int t = 0; t < E.T; t += step) if (tot[(size_t) t] > thr) keep.push_back (t); }
    if (keep.size() < 4)        // next to nothing sounds: the four loudest frames
    {
        Vec s; for (int t = 0; t < E.T; t += step) s.push_back (tot[(size_t) t]);
        std::sort (s.begin(), s.end()); const double least = s[s.size() - std::min<size_t> (4, s.size())];
        keep.clear(); for (int t = 0; t < E.T; t += step) if (tot[(size_t) t] >= least) keep.push_back (t);
    }
    const size_t K = keep.size(); double mt = 0; for (int t : keep) mt += tot[(size_t) t]; mt /= (double) K;
    const double ref = 10.0 * std::log10 (mt + 1e-30);
    C.c.assign ((size_t) CURVES * (size_t) NBAND, 0.0f); C.frames = (int) K; Vec d (K);
    auto place = [K] (double p) { return (size_t) std::nearbyint (p * (double) (K - 1)); };        // the frame at this place in the order (halves go to the even one, as numpy rounds)
    for (int b = 0; b < NBAND; ++b)
    {
        double mean = 0;
        for (size_t i = 0; i < K; ++i) { const double v = E.row (keep[i])[b] * g[(size_t) b]; mean += v; d[i] = 10.0 * std::log10 (v + 1e-30) - ref; }
        std::sort (d.begin(), d.end());
        const double v[CURVES] = { 10.0 * std::log10 (mean / (double) K + 1e-30) - ref, d[0], d[place (0.1)], d[place (0.5)], d[place (0.9)], d[K - 1] };
        for (int c = 0; c < CURVES; ++c) C.c[(size_t) c * (size_t) NBAND + (size_t) b] = (float) std::max (v[c], FLOOR);
    }
    return C;
}

// ---------------------------------------------------------------- the net
struct Net
{
    struct Layer { int in = 0, out = 0, k = 1, dil = 1; std::vector<float> w, b; };     // w[out][in][k]
    std::vector<Layer> conv; Layer curve, there;        // there: one number from every channel's mean and its maximum over the bands
    bool ready() const { return ! conv.empty(); }
    [[noreturn]] static void fail (const char* why) { throw std::runtime_error (std::string ("curve reader model: ") + why); }

    // One net's block of the file written by Tools/export_reader_model.py, from a cursor that moves on. Throws
    // std::runtime_error when the block is not complete or not of the sizes its own header gives.
    static Net parse (const uint8_t*& p, size_t& left)
    {
        auto take = [&] (void* dst, size_t bytes) { if (bytes > left) fail ("the file is cut short"); std::memcpy (dst, p, bytes); p += bytes; left -= bytes; };
        auto u32 = [&] { uint32_t v = 0; take (&v, 4); return v; };
        auto floats = [&] (std::vector<float>& v, size_t count)
        {
            if ((size_t) u32() != count) fail ("a layer does not have the size the header gives");
            if (count > left / 4) fail ("the file is cut short");
            v.resize (count); take (v.data(), count * 4);
            for (float f : v) if (! std::isfinite (f)) fail ("a number is not finite");
        };
        auto layer = [&] (int in, int out, int k, int dil) { Layer l; l.in = in; l.out = out; l.k = k; l.dil = dil; floats (l.w, (size_t) out * (size_t) in * (size_t) k); floats (l.b, (size_t) out); return l; };
        if (u32() != (uint32_t) (CURVES + 1)) fail ("made for a different number of input curves");
        if (u32() != (uint32_t) NBAND) fail ("made for a different number of bands");
        const uint32_t layers = u32(), ch = u32(), k = u32();
        if (layers < 1 || layers > 16) fail ("bad number of layers");
        if (ch < 1 || ch > 512) fail ("bad channel count");
        if (k < 1 || k > 15 || (k & 1) == 0) fail ("bad width");
        Net m;
        for (uint32_t i = 0; i < layers; ++i) { const uint32_t dil = u32(); if (dil < 1 || dil > 64) fail ("bad spacing"); m.conv.push_back (layer (i == 0 ? CURVES + 1 : (int) ch, (int) ch, (int) k, (int) dil)); }
        m.curve = layer ((int) ch, 1, 1, 1); m.there = layer (2 * (int) ch, 1, 1, 1);
        return m;
    }

    // The reader's answer for one sound's curves: the curve in dB per band, and how sure it is (a logit: above 0 = more
    // likely there than not).
    void run (const Curves& C, Vec& out, double& sure) const
    {
        out.assign ((size_t) NBAND, 0.0); sure = -99.0; if (! ready() || C.c.size() != (size_t) CURVES * (size_t) NBAND) return;
        std::vector<float> a ((size_t) (CURVES + 1) * (size_t) NBAND), b2;
        for (int c = 0; c < CURVES; ++c) for (int b = 0; b < NBAND; ++b) a[(size_t) c * (size_t) NBAND + (size_t) b] = C.at (c, b) / 20.0f + 2.0f;
        for (int b = 0; b < NBAND; ++b) a[(size_t) CURVES * (size_t) NBAND + (size_t) b] = (float) (-1.0 + 2.0 * (double) b / (double) (NBAND - 1));       // the band's place
        for (const Layer& L : conv)
        {
            b2.assign ((size_t) L.out * (size_t) NBAND, 0.0f); const int half = L.k / 2;
            for (int o = 0; o < L.out; ++o)
            {
                float* dst = b2.data() + (size_t) o * (size_t) NBAND; for (int b = 0; b < NBAND; ++b) dst[b] = L.b[(size_t) o];
                for (int i = 0; i < L.in; ++i)
                {
                    const float* src = a.data() + (size_t) i * (size_t) NBAND; const float* w = L.w.data() + ((size_t) o * (size_t) L.in + (size_t) i) * (size_t) L.k;
                    for (int j = 0; j < L.k; ++j)
                    {
                        const int sh = (j - half) * L.dil; const float wj = w[j]; const int b0 = std::max (0, -sh), b1 = std::min (NBAND, NBAND - sh);
                        for (int b = b0; b < b1; ++b) dst[b] += wj * src[b + sh];
                    }
                }
                for (int b = 0; b < NBAND; ++b) dst[b] = std::max (dst[b], 0.0f);
            }
            a.swap (b2);
        }
        const int ch = conv.back().out; double s = there.b[0];
        for (int b = 0; b < NBAND; ++b) { double v = curve.b[0]; for (int c = 0; c < ch; ++c) v += (double) curve.w[(size_t) c] * (double) a[(size_t) c * (size_t) NBAND + (size_t) b]; out[(size_t) b] = 20.0 * v; }
        for (int c = 0; c < ch; ++c)
        {
            const float* src = a.data() + (size_t) c * (size_t) NBAND; double mean = 0; float mx = src[0];
            for (int b = 0; b < NBAND; ++b) { mean += src[b]; mx = std::max (mx, src[b]); }
            s += (double) there.w[(size_t) c] * mean / (double) NBAND + (double) there.w[(size_t) ch + (size_t) c] * (double) mx;
        }
        sure = s;
    }
};

// Several nets taught the same thing, each from a start of its own, read together. Band by band the answer is the
// one nearest to zero, and zero where they do not agree on the sign: what only one of them reads is not taken. (A cut
// that is read too deep does harm when it is undone, one read too shallow leaves some of the filter in.)
struct Reader
{
    std::vector<Net> nets;
    bool ready() const { return ! nets.empty(); }

    // Reads the file written by Tools/export_reader_model.py from memory. Throws std::runtime_error, and stays as it
    // was, when the block is not a complete model file.
    void load (const void* data, size_t size)
    {
        const uint8_t* p = (const uint8_t*) data; size_t left = data != nullptr ? size : 0;
        if (left < 12 || std::memcmp (p, "DRCR", 4) != 0) Net::fail ("not a curve reader model file");
        uint32_t version = 0, count = 0; std::memcpy (&version, p + 4, 4); std::memcpy (&count, p + 8, 4); p += 12; left -= 12;
        if (version != 2) Net::fail ("unknown format version");
        if (count < 1 || count > 16) Net::fail ("bad number of nets");
        std::vector<Net> m; for (uint32_t i = 0; i < count; ++i) m.push_back (Net::parse (p, left));
        if (left != 0) Net::fail ("more data than the header gives");
        nets = std::move (m);
    }

    // The nets' answer for one sound's curves: the curve (dB per band; the least of what they read), the mean of how
    // sure they are (a logit: above 0 = more likely there than not), and per band how far the nets lie apart (dB,
    // largest minus smallest).
    void run (const Curves& C, Vec& out, double& sure, Vec* apart = nullptr) const
    {
        out.assign ((size_t) NBAND, 0.0); sure = -99.0; if (apart != nullptr) apart->assign ((size_t) NBAND, 0.0);
        if (! ready()) return;
        Vec one, lo ((size_t) NBAND, 1e9), hi ((size_t) NBAND, -1e9); double s = 0, s1 = 0;
        for (const Net& n : nets)
        {
            n.run (C, one, s1); s += s1;
            for (int b = 0; b < NBAND; ++b) { const double v = one[(size_t) b]; lo[(size_t) b] = std::min (lo[(size_t) b], v); hi[(size_t) b] = std::max (hi[(size_t) b], v); }
        }
        sure = s / (double) nets.size();
        for (int b = 0; b < NBAND; ++b) out[(size_t) b] = lo[(size_t) b] > 0.0 ? lo[(size_t) b] : hi[(size_t) b] < 0.0 ? hi[(size_t) b] : 0.0;
        if (apart != nullptr) for (int b = 0; b < NBAND; ++b) (*apart)[(size_t) b] = hi[(size_t) b] - lo[(size_t) b];
    }
};

// ---------------------------------------------------------------- a curve over the bands as a filter
// The curve at any frequency: straight lines between the bands on a log-frequency axis; beyond the first and the
// last band it runs on with the slope it has over the half octave there (a filter's skirt goes on), never upwards.
inline double atHz (const Vec& band, double hz)
{
    const double lf = std::log2 (std::max (hz, 1.0)), l0 = std::log2 (F0), pos = (lf - l0) * 12.0;
    if (pos <= 0.0) { const double slope = (band[6] - band[0]) / 0.5; return band[0] - std::max (slope, 0.0) * (l0 - lf); }
    if (pos >= (double) (NBAND - 1)) { const double slope = (band[(size_t) NBAND - 1] - band[(size_t) NBAND - 7]) / 0.5; return band[(size_t) NBAND - 1] + std::min (slope, 0.0) * (lf - (l0 + (double) (NBAND - 1) / 12.0)); }
    const int i = (int) pos; const double t = pos - (double) i; return band[(size_t) i] + (band[(size_t) i + 1] - band[(size_t) i]) * t;
}
// The curve as a minimum-phase impulse response of n taps, held between lo and hi (dB): the log magnitude on a fine
// frequency grid, its cepstrum folded (as tone::toIr does it; here the curve is asked at every grid frequency, so a
// steep skirt at the low end keeps its shape).
inline Vec toIr (const Vec& band, double lo, double hi, int n = tone::irTaps)
{
    const int nfft = (int) nextPow2 ((size_t) 4 * (size_t) n); const room::Transform tr (nfft); Vec re ((size_t) nfft, 0.0), im ((size_t) nfft, 0.0);
    const double ln = std::log (10.0) / 20.0;
    for (int j = 0; j <= nfft / 2; ++j)
    {
        re[(size_t) j] = std::clamp (atHz (band, std::max ((double) j * SR / (double) nfft, 1.0)), lo, hi) * ln;
        if (j > 0 && j < nfft / 2) re[(size_t) (nfft - j)] = re[(size_t) j];
    }
    tr.back (re, im);                                                       // the cepstrum (real, even)
    for (int j = 1; j < nfft / 2; ++j) re[(size_t) j] *= 2.0;
    for (int j = nfft / 2 + 1; j < nfft; ++j) re[(size_t) j] = 0.0;
    std::fill (im.begin(), im.end(), 0.0); tr.run (re, im);
    for (int j = 0; j <= nfft / 2; ++j)
    {
        const double m = std::exp (re[(size_t) j]), ph = im[(size_t) j]; const bool edge = j == 0 || j == nfft / 2;
        const double er = m * std::cos (ph), ei = edge ? 0.0 : m * std::sin (ph);
        re[(size_t) j] = er; im[(size_t) j] = ei; if (! edge) { re[(size_t) (nfft - j)] = er; im[(size_t) (nfft - j)] = -ei; }
    }
    tr.back (re, im);
    Vec h (re.begin(), re.begin() + n); const int kf = std::max (8, n / 8);
    for (int i = 0; i < kf && i < n; ++i) h[(size_t) (n - kf + i)] *= 0.5 * (1.0 + std::cos (PI * ((double) i + 0.5) / (double) kf));
    return h;
}
// A curve moved along the frequency axis by `octaves` (up when positive); what comes in from beyond the ends is the
// end's own value.
inline Vec moved (const Vec& band, double octaves)
{
    if (octaves == 0.0) return band;
    Vec o ((size_t) NBAND);
    for (int b = 0; b < NBAND; ++b)
    {
        const double pos = std::clamp ((double) b - 12.0 * octaves, 0.0, (double) (NBAND - 1)); const int i = std::min ((int) pos, NBAND - 2); const double t = pos - (double) i;
        o[(size_t) b] = band[(size_t) i] + (band[(size_t) i + 1] - band[(size_t) i]) * t;
    }
    return o;
}
// A sound through a curve over the bands (dB), as a minimum-phase filter; held between lo and hi. Empty when given up.
inline Stereo through (const Stereo& x, const Vec& band, double lo, double hi, const Stop& stop = {})
{
    bool any = false; for (double v : band) if (v != 0.0) any = true;
    if (! any || band.size() != (size_t) NBAND) return x;
    const Vec h = toIr (band, lo, hi); return tone::filter (x, h, h, stop);
}

} // namespace reader
} // namespace dr
