#pragma once
// Which effects are on a sound, and with what settings (port of inspector.py + read_settings).

#include "Hand.h"
#include "Models.h"
#include <functional>

namespace dr
{

struct Inspection
{
    std::map<std::string, double> prob;                                   // effect -> 0..1 (max over windows)
    std::map<std::string, std::map<std::string, double>> settings;        // effect -> parameter -> value (categories as index)
    std::map<std::string, std::map<std::string, std::string>> labels;     // category parameters as text
    std::vector<std::string> found (double cut = 0.65) const
    {
        std::vector<std::string> f; for (auto& [k, v] : prob) if (v > cut) f.push_back (k); return f;
    }
};

inline std::vector<Stereo> windows3s (const Stereo& y)
{
    const size_t W = (size_t) (3 * SR); std::vector<Stereo> out;
    if (y.size() <= W) { out.push_back (y.slice (0, W)); return out; }
    for (size_t s = 0; s + W <= y.size(); s += W / 2) out.push_back (y.slice (s, W));
    return out;
}

inline Inspection inspect (const Models& M, const Stereo& y, const std::function<bool (float)>& progress = {})
{
    Inspection I;
    const auto wins = windows3s (y);
    std::vector<std::vector<float>> P, rows;
    size_t done = 0;
    for (const auto& w : wins)
    {
        ++done;
        double e = 0; for (size_t i = 0; i < w.size(); ++i) e += 0.5 * (w.l[i] * w.l[i] + w.r[i] * w.r[i]);
        if (std::sqrt (e / (double) w.size()) < 1e-4) continue;
        const auto o = M.run (features (w));
        const auto h = handFeatures (w);
        std::vector<float> x = o.logits; x.insert (x.end(), h.begin(), h.end());
        std::vector<float> p; for (auto& d : M.decide) p.push_back ((float) d.prob (x));
        P.push_back (p);
        std::vector<float> z = o.emb; z.insert (z.end(), h.begin(), h.end()); rows.push_back (z);
        if (progress && ! progress ((float) done / (float) wins.size())) return I;
    }
    for (size_t k = 0; k < M.fx.size(); ++k)
    {
        double m = 0; for (auto& p : P) m = std::max (m, (double) p[k]);
        I.prob[M.fx[k]] = m;
    }
    // whole-file echo check: a long file gives the echo finder far more evidence than one window
    const auto [sc, ms] = hand::echoDetect (y);
    (void) ms;
    if (sc > 60 && I.prob["bitcrush"] < 0.5) I.prob["delay"] = std::max (I.prob["delay"], std::min (0.99, 0.6 + sc / 1000.0));
    // settings readers: median over the windows
    for (auto& [fx, params] : M.settings)
        for (auto& [k, s] : params)
        {
            if (rows.empty()) continue;
            if (s.kind == "cat")
            {
                std::vector<int> votes (s.cats.size(), 0);
                for (auto& r : rows) ++votes[s.m.argmax (r)];
                const size_t best = (size_t) (std::max_element (votes.begin(), votes.end()) - votes.begin());
                I.settings[fx][k] = (double) best; I.labels[fx][k] = s.cats[best];
            }
            else
            {
                Vec v; for (auto& r : rows) v.push_back (s.m.raw (r)[0]);
                const double med = median (v);
                I.settings[fx][k] = s.kind == "log" ? std::exp (med) : med;
            }
        }
    return I;
}

} // namespace dr
