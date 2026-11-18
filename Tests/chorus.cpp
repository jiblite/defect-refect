// The chorus engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/chorus_parity.py):
//   drchorus a.wav b.wav prefix
// reads the voices of a chorus off a, prints what it found as one line of JSON, and writes as raw float64 files
//   prefix.peaks0.f64, prefix.peaks1.f64     the peaks of the left and of the right side: rows of frame, delay, stretch, height
//   prefix.out.f64     a with the voices taken out (apply s = -1), left then right
//   prefix.half.f64    a with the voices half taken out (s = -0.5)
//   prefix.onto.f64    b with the voices put on (s = +1);  prefix.onto2.f64  twice as strong (s = +2)
//   prefix.fast.f64    b with the voices put on, moving 1.5 times as fast and half as far
//   prefix.slow.f64    a with its own voices moving half as fast and 1.5 times as far (s = 0)
// and the time each step took. Nothing found: writes only the peaks.
//   drchorus both a.wav b.wav outA.f64 outB.f64
// the engine as the app uses it, on a sound of any length: a with its chorus taken out, a's chorus put on b (raw float64,
// left then right; written only if a chorus was found), and one line of JSON about what was found.
//   drchorus find a.wav ...
// only what is found on each sound, one line of JSON each.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Chorus.h"

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
static void say (const dr::chorus::Found& F, double took, int p0 = -1, int p1 = -1)
{
    std::printf ("{\"took\": %.3f, \"found\": %s, \"sure\": %.6f, \"n\": %zu, \"peaks\": [%d, %d], \"pieces\": [", took, F.found ? "true" : "false", F.sure, F.n, p0, p1);
    for (size_t k = 0; k < F.pieces.size(); ++k)
    {
        const auto& pc = F.pieces[k]; std::printf ("%s{\"a\": %zu, \"n\": %zu, \"peaked\": [[%.6g, %.6g], [%.6g, %.6g]], \"voices\": [", k ? ", " : "", pc.a, pc.n, pc.peaked[0][0], pc.peaked[0][1], pc.peaked[1][0], pc.peaked[1][1]);
        for (int c = 0; c < 2; ++c)
        {
            std::printf ("%s[", c ? ", " : "");
            for (size_t i = 0; i < pc.voices[c].size(); ++i)
            {
                const auto& v = pc.voices[c][i];
                std::printf ("%s{\"kind\": %d, \"rate\": %.9g, \"phase\": %.9g, \"mid\": %.9g, \"half\": %.9g, \"on\": %.6g, \"height\": %.6g, \"piled\": %.6g, \"loud\": %.6g, \"before\": %.6g, \"after\": %.6g, \"least\": %.6g, \"twin\": %.6g, \"clear\": %.6g, \"ahead\": %d, \"lean\": %.6g, \"cover\": [%.3g, %.3g, %.3g, %.3g, %.3g, %.3g, %.3g, %.3g], \"gain\": [", i ? ", " : "", v.kind, v.rate, v.phase, v.mid, v.half, v.on, v.height, v.piled, v.loud, v.before, v.after, v.least, v.twin, dr::chorus::clear (v), v.ahead ? 1 : 0, v.lean, v.cover[0], v.cover[1], v.cover[2], v.cover[3], v.cover[4], v.cover[5], v.cover[6], v.cover[7]);
                for (int b = 0; b < dr::chorus::NB; ++b) std::printf ("%s%.6g", b ? ", " : "", v.gain[b]);
                std::printf ("]}");
            }
            std::printf ("]");
        }
        std::printf ("]}");
    }
    std::printf ("], \"notes\": [");
    for (size_t i = 0; i < F.notes.size(); ++i) std::printf ("%s\"%s\"", i ? ", " : "", F.notes[i].c_str());
    std::printf ("]}\n");
}
int main (int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock; namespace ch = dr::chorus;
        if (argc >= 6 && std::string (argv[1]) == "both")      // as the app uses the engine: a without its chorus, a's chorus on b; one line of JSON
        {
            const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); const auto t0 = clock::now();
            const ch::Found F = ch::find (a); const double took = std::chrono::duration<double> (clock::now() - t0).count();
            if (F.found)
            {
                const dr::Stereo out = ch::apply (a, F, -1.0); writeRaw (argv[4], { &out.l, &out.r });
                const dr::Stereo onto = ch::apply (b, F, 1.0, 1.0, 1.0, false); writeRaw (argv[5], { &onto.l, &onto.r });
            }
            say (F, took); return 0;
        }
        if (argc >= 3 && std::string (argv[1]) == "find")      // only what is found: one line of JSON for each sound
        {
            for (int i = 2; i < argc; ++i) { const dr::Stereo a = sound (argv[i]); const auto t0 = clock::now(); const ch::Found F = ch::find (a); say (F, std::chrono::duration<double> (clock::now() - t0).count()); std::fflush (stdout); }
            return 0;
        }
        if (argc < 4) { std::cout << "drchorus a.wav b.wav prefix | drchorus both a.wav b.wav outA.f64 outB.f64 | drchorus find a.wav ...\n"; return 1; }
        const dr::Stereo a = sound (argv[1]), b = sound (argv[2]); const std::string pre = argv[3];
        auto t0 = clock::now();
        auto lap = [&] (const char* what) { const auto t1 = clock::now(); std::fprintf (stderr, "  %s: %.2f s\n", what, std::chrono::duration<double> (t1 - t0).count()); t0 = t1; };
        int np[2] = { 0, 0 };
        for (int c = 0; c < 2; ++c)
        {
            const ch::Peaks P = ch::peaksOf (ch::framesOf (c == 0 ? a.l : a.r)); dr::Vec rows; np[c] = (int) P.size();
            for (auto& p : P) { rows.push_back (p.t); rows.push_back (p.d); rows.push_back (p.r); rows.push_back (p.h); }
            writeRaw (pre + (c == 0 ? ".peaks0.f64" : ".peaks1.f64"), { &rows });
        }
        lap ("peaks (both sides, again inside find)");
        const auto start = clock::now(); const ch::Found F = ch::find (a); const double took = std::chrono::duration<double> (clock::now() - start).count(); lap ("find");
        say (F, took, np[0], np[1]);
        if (! F.found) return 0;
        const dr::Stereo out = ch::apply (a, F, -1.0); lap ("taken out"); writeRaw (pre + ".out.f64", { &out.l, &out.r });
        const dr::Stereo half = ch::apply (a, F, -0.5); writeRaw (pre + ".half.f64", { &half.l, &half.r });
        const dr::Stereo onto = ch::apply (b, F, 1.0, 1.0, 1.0, false); lap ("put on"); writeRaw (pre + ".onto.f64", { &onto.l, &onto.r });
        const dr::Stereo onto2 = ch::apply (b, F, 2.0, 1.0, 1.0, false); writeRaw (pre + ".onto2.f64", { &onto2.l, &onto2.r });
        const dr::Stereo fast = ch::apply (b, F, 1.0, 1.5, 0.5, false); writeRaw (pre + ".fast.f64", { &fast.l, &fast.r });
        const dr::Stereo slow = ch::apply (a, F, 0.0, 0.5, 1.5, true); lap ("moved slower and further"); writeRaw (pre + ".slow.f64", { &slow.l, &slow.r });
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drchorus: " << e.what() << "\n"; return 1; }
}
