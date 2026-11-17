// The moving-tone engine (phaser, filter sweep) without the UI (48 kHz WAV in), for comparison with the Python prototype
// (Tools/moving_parity.py):
//   drmoving model.bin a.wav b.wav prefix
// reads the moving tone curve off a, prints what it found, and writes as raw float64 files
//   prefix.g.f64       the model's curve, left then right, each [frame][band] in dB
//   prefix.on.f64      per frame: the model's say-so that a moving filter is at work
//   prefix.out.f64     a with the curve taken off (apply s = -1), left then right
//   prefix.half.f64    a with the curve half taken off (s = -0.5)
//   prefix.onto.f64    b with the curve put on (s = +1);  prefix.onto2.f64  twice as strong (s = +2)
//   prefix.fast.f64    b with the curve put on, moving 1.5 times as fast
//   prefix.slow.f64    a with its own curve moving half as fast (s = 0, speed 0.5)
// and the time each step took. Nothing found: writes only the first two.
//   drmoving both model.bin a.wav b.wav outA.f64 outB.f64
// the engine as the app uses it, on a sound of any length: a with its curve taken off, a's curve put on b (raw float64,
// left then right; written only if a moving filter was found), and one line of JSON about what was found.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Moving.h"

static void writeRaw (const std::string& p, const std::vector<const dr::Vec*>& vs)
{
    std::ofstream f (p, std::ios::binary);
    for (auto* v : vs) f.write ((const char*) v->data(), (std::streamsize) (v->size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    if (std::abs (sr - 48000.0) > 0.5) throw std::runtime_error (path + " is not at 48 kHz");
    return y;
}
static dr::moving::Net model (const std::string& path)
{
    std::ifstream f (path, std::ios::binary); if (! f) throw std::runtime_error ("cannot read " + path);
    const std::vector<char> blob ((std::istreambuf_iterator<char> (f)), std::istreambuf_iterator<char>());
    dr::moving::Net net; net.load (blob.data(), blob.size()); return net;
}
static void say (const dr::moving::Found& F, double took)
{
    std::printf ("{\"took\": %.3f, \"found\": %s, \"kind\": \"%s\", \"sure\": %.6f, \"period\": %.4f, \"frames\": %d, \"spans\": [", took, F.found ? "true" : "false", F.kind.c_str(), F.sure, F.period, F.T);
    for (size_t i = 0; i < F.spans.size(); ++i) std::printf ("%s[%d, %d]", i ? ", " : "", F.spans[i].first, F.spans[i].second);
    std::printf ("]}\n");
}
int main (int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock;
        if (argc >= 7 && std::string (argv[1]) == "both")      // as the app uses the engine: a without its moving filter, a's filter on b; one line of JSON
        {
            const dr::moving::Net net = model (argv[2]); const dr::Stereo a = sound (argv[3]), b = sound (argv[4]); const auto t0 = clock::now();
            const dr::moving::Found F = dr::moving::find (net, a); const double took = std::chrono::duration<double> (clock::now() - t0).count();
            if (F.found)
            {
                const dr::Stereo out = dr::moving::apply (a, F, -1.0); writeRaw (argv[5], { &out.l, &out.r });
                const dr::Stereo onto = dr::moving::apply (b, F, 1.0, 1.0, false); writeRaw (argv[6], { &onto.l, &onto.r });
            }
            say (F, took); return 0;
        }
        if (argc < 5) { std::cout << "drmoving model.bin a.wav b.wav prefix | drmoving both model.bin a.wav b.wav outA.f64 outB.f64\n"; return 1; }
        const dr::moving::Net net = model (argv[1]); const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); const std::string pre = argv[4];
        auto t0 = clock::now(); const auto start = t0;
        auto lap = [&] (const char* what) { const auto t1 = clock::now(); std::printf ("  %s: %.2f s\n", what, std::chrono::duration<double> (t1 - t0).count()); t0 = t1; };
        const dr::moving::Found F = dr::moving::find (net, a); const double took = std::chrono::duration<double> (clock::now() - start).count(); lap ("find");
        say (F, took);
        const dr::Vec gl (F.raw[0].begin(), F.raw[0].end()), gr (F.raw[1].begin(), F.raw[1].end());
        writeRaw (pre + ".g.f64", { &gl, &gr }); writeRaw (pre + ".on.f64", { &F.on });
        if (! F.found) return 0;
        const dr::Stereo out = dr::moving::apply (a, F, -1.0); lap ("taken off"); writeRaw (pre + ".out.f64", { &out.l, &out.r });
        const dr::Stereo half = dr::moving::apply (a, F, -0.5); writeRaw (pre + ".half.f64", { &half.l, &half.r });
        const dr::Stereo onto = dr::moving::apply (b, F, 1.0, 1.0, false); lap ("put on"); writeRaw (pre + ".onto.f64", { &onto.l, &onto.r });
        const dr::Stereo onto2 = dr::moving::apply (b, F, 2.0, 1.0, false); writeRaw (pre + ".onto2.f64", { &onto2.l, &onto2.r });
        const dr::Stereo fast = dr::moving::apply (b, F, 1.0, 1.5, false); writeRaw (pre + ".fast.f64", { &fast.l, &fast.r });
        const dr::Stereo slow = dr::moving::apply (a, F, 0.0, 0.5, true); writeRaw (pre + ".slow.f64", { &slow.l, &slow.r });
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drmoving: " << e.what() << "\n"; return 1; }
}
