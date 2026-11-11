#pragma once
// The trained models: the neural detector (a small CNN), the second-stage effect deciders and
// the settings readers (gradient-boosted trees). Loaded from Resources/models.bin, written by
// Tools/export_models.py. Plain C++17.

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

#include "Features.h"

namespace dr
{

class Reader
{
public:
    Reader (const void* d, size_t n) : p ((const uint8_t*) d), end ((const uint8_t*) d + n) {}
    uint32_t u32() { uint32_t v; take (&v, 4); return v; }
    int32_t i32() { int32_t v; take (&v, 4); return v; }
    float f32() { float v; take (&v, 4); return v; }
    uint8_t u8() { uint8_t v; take (&v, 1); return v; }
    std::vector<float> f32a() { const uint32_t n = u32(); std::vector<float> v (n); take (v.data(), (size_t) n * 4); return v; }
    std::string str() { const uint32_t n = u32(); std::string s (n, '\0'); take (s.data(), n); return s; }
    void take (void* dst, size_t n) { if (p + n > end) throw std::runtime_error ("models.bin is truncated"); std::memcpy (dst, p, n); p += n; }
private:
    const uint8_t* p; const uint8_t* end;
};

// ---------------------------------------------------------------- gradient-boosted trees
struct Node { int32_t feat; float thr; int32_t left, right; uint8_t leaf, missLeft; float value; };
struct Boosted
{
    std::vector<float> base;                         // one per output (1, or classes for multi-class)
    std::vector<std::vector<Node>> trees;            // iteration-major: trees[it * nOut + k]
    void load (Reader& r)
    {
        base = r.f32a(); const uint32_t iters = r.u32();
        trees.resize ((size_t) iters * base.size());
        for (auto& t : trees)
        {
            t.resize (r.u32());
            for (auto& nd : t) { nd.feat = r.i32(); nd.thr = r.f32(); nd.left = r.i32(); nd.right = r.i32(); nd.leaf = r.u8(); nd.missLeft = r.u8(); nd.value = r.f32(); }
        }
    }
    std::vector<double> raw (const std::vector<float>& x) const
    {
        const size_t K = base.size(); std::vector<double> o (base.begin(), base.end());
        for (size_t i = 0; i < trees.size(); ++i)
        {
            const auto& t = trees[i]; size_t n = 0;
            while (! t[n].leaf)
            {
                const float v = x[(size_t) t[n].feat];
                n = (size_t) (std::isnan (v) ? (t[n].missLeft ? t[n].left : t[n].right) : (v <= t[n].thr ? t[n].left : t[n].right));
            }
            o[i % K] += t[n].value;
        }
        return o;
    }
    double prob (const std::vector<float>& x) const { return 1.0 / (1.0 + std::exp (-raw (x)[0])); }
    size_t argmax (const std::vector<float>& x) const { const auto r = raw (x); return (size_t) (std::max_element (r.begin(), r.end()) - r.begin()); }
};

// ---------------------------------------------------------------- CNN
struct Conv { int out = 0, in = 0, k = 3, pad = 1, dil = 1; std::vector<float> w, b; };
struct Linear { int out = 0, in = 0; std::vector<float> w, b; };

struct Models
{
    std::vector<Conv> conv2;                // 10 convs (2 per block), BN folded
    std::vector<std::pair<int, int>> pools; // per block (freq, time)
    Conv t1, t2;                            // temporal 1-D convs
    Linear cls, reg;
    std::vector<std::string> fx;            // effect names, detector order
    std::vector<Boosted> decide;            // per effect: second-stage decider
    struct Setting { std::string kind; std::vector<std::string> cats; Boosted m; };
    std::map<std::string, std::map<std::string, Setting>> settings; // effect -> parameter -> reader

    void load (const void* data, size_t n)
    {
        Reader r (data, n); char mg[4]; r.take (mg, 4);
        if (std::memcmp (mg, "DRFX", 4) != 0) throw std::runtime_error ("models.bin: bad header");
        r.u32();
        conv2.resize (r.u32());
        for (auto& c : conv2) { c.out = (int) r.u32(); c.in = (int) r.u32(); c.w = r.f32a(); c.b = r.f32a(); }
        pools.resize (r.u32()); for (auto& p : pools) { p.first = (int) r.u32(); p.second = (int) r.u32(); }
        for (Conv* c : { &t1, &t2 }) { c->out = (int) r.u32(); c->in = (int) r.u32(); c->k = (int) r.u32(); c->pad = (int) r.u32(); c->dil = (int) r.u32(); c->w = r.f32a(); c->b = r.f32a(); }
        for (Linear* l : { &cls, &reg }) { l->out = (int) r.u32(); l->in = (int) r.u32(); l->w = r.f32a(); l->b = r.f32a(); }
        fx.resize (r.u32()); for (auto& s : fx) s = r.str();
        const uint32_t nd = r.u32(); decide.resize (nd);
        for (uint32_t i = 0; i < nd; ++i) { r.str(); decide[i].load (r); }
        const uint32_t ns = r.u32();
        for (uint32_t i = 0; i < ns; ++i)
        {
            const std::string f = r.str(); const uint32_t np = r.u32();
            for (uint32_t j = 0; j < np; ++j)
            {
                const std::string k = r.str(); Setting s; s.kind = r.str();
                const uint32_t nc = r.u32(); for (uint32_t c = 0; c < nc; ++c) s.cats.push_back (r.str());
                s.m.load (r); settings[f][k] = std::move (s);
            }
        }
    }

    struct Out { std::vector<float> logits, reg, emb; };

    // Forward pass of the detector for one feature map (4 x 256 x T).
    Out run (const FeatureMap& F) const
    {
        const int T = F.frames; Out o;
        // average neighbouring bands: 256 -> 128
        int C = 4, H = 128, W = T;
        std::vector<float> x ((size_t) C * H * W);
        for (int c = 0; c < C; ++c) for (int h = 0; h < H; ++h) for (int t = 0; t < W; ++t)
            x[((size_t) c * H + h) * W + t] = 0.5f * (F.at (c, 2 * h, t) + F.at (c, 2 * h + 1, t));
        for (size_t blk = 0; blk < pools.size(); ++blk)
        {
            for (int j = 0; j < 2; ++j) { x = conv2d (conv2[blk * 2 + (size_t) j], x, C, H, W); C = conv2[blk * 2 + (size_t) j].out; }
            const int pf = pools[blk].first, pt = pools[blk].second, H2 = H / pf, W2 = W / pt;
            std::vector<float> y ((size_t) C * H2 * W2);
            for (int c = 0; c < C; ++c) for (int h = 0; h < H2; ++h) for (int t = 0; t < W2; ++t)
            {
                float m = -1e30f;
                for (int a = 0; a < pf; ++a) for (int b = 0; b < pt; ++b) m = std::max (m, x[((size_t) c * H + (h * pf + a)) * W + (t * pt + b)]);
                y[((size_t) c * H2 + h) * W2 + t] = m;
            }
            x.swap (y); H = H2; W = W2;
        }
        // (C, H, W) -> (C*H, W) channels-first for the 1-D convs
        int ch = C * H;
        x = conv1d (t1, x, ch, W); ch = t1.out;
        x = conv1d (t2, x, ch, W); ch = t2.out;
        o.emb.resize ((size_t) 2 * ch);
        for (int c = 0; c < ch; ++c)
        {
            double s = 0; float m = -1e30f;
            for (int t = 0; t < W; ++t) { s += x[(size_t) c * W + t]; m = std::max (m, x[(size_t) c * W + t]); }
            o.emb[(size_t) c] = (float) (s / std::max (1, W)); o.emb[(size_t) (ch + c)] = m;
        }
        o.logits = linear (cls, o.emb); o.reg = linear (reg, o.emb);
        return o;
    }

private:
    static std::vector<float> conv2d (const Conv& k, const std::vector<float>& x, int C, int H, int W)
    {
        std::vector<float> y ((size_t) k.out * H * W);
        for (int o = 0; o < k.out; ++o)
        {
            float* yo = &y[(size_t) o * H * W];
            std::fill (yo, yo + (size_t) H * W, k.b[(size_t) o]);
            for (int c = 0; c < C; ++c)
            {
                const float* xc = &x[(size_t) c * H * W];
                for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
                {
                    const float w = k.w[(((size_t) o * C + c) * 3 + a) * 3 + b];
                    for (int h = 0; h < H; ++h)
                    {
                        const int hh = h + a - 1; if (hh < 0 || hh >= H) continue;
                        const float* xr = xc + (size_t) hh * W; float* yr = yo + (size_t) h * W;
                        const int t0 = std::max (0, 1 - b), t1e = std::min (W, W + 1 - b);
                        for (int t = t0; t < t1e; ++t) yr[t] += w * xr[t + b - 1];
                    }
                }
            }
            for (size_t i = 0; i < (size_t) H * W; ++i) yo[i] = std::max (0.0f, yo[i]);
        }
        return y;
    }
    static std::vector<float> conv1d (const Conv& k, const std::vector<float>& x, int C, int W)
    {
        std::vector<float> y ((size_t) k.out * W);
        for (int o = 0; o < k.out; ++o)
        {
            float* yo = &y[(size_t) o * W];
            std::fill (yo, yo + W, k.b[(size_t) o]);
            for (int c = 0; c < C; ++c)
            {
                const float* xc = &x[(size_t) c * W];
                for (int a = 0; a < k.k; ++a)
                {
                    const float w = k.w[((size_t) o * C + c) * k.k + a];
                    const int off = a * k.dil - k.pad;
                    for (int t = std::max (0, -off); t < std::min (W, W - off); ++t) yo[t] += w * xc[t + off];
                }
            }
            for (int t = 0; t < W; ++t) yo[t] = std::max (0.0f, yo[t]);
        }
        return y;
    }
    static std::vector<float> linear (const Linear& l, const std::vector<float>& x)
    {
        std::vector<float> y ((size_t) l.out);
        for (int o = 0; o < l.out; ++o) { double s = l.b[(size_t) o]; for (int i = 0; i < l.in; ++i) s += (double) l.w[(size_t) o * l.in + i] * x[(size_t) i]; y[(size_t) o] = (float) s; }
        return y;
    }
};

} // namespace dr
