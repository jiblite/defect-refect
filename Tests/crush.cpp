// The crush engine without the UI (WAV in), for comparison with the Python prototype (Tools/crush_parity.py):
//   drcrush a.wav b.wav prefix
// reads the grid and the hold off a, prints what it found as one line of JSON, and writes as raw float64 (left then right)
//   prefix.onto.f64     b through the found crusher (Amount 1);  prefix.onto2.f64  Amount 2;  prefix.onto05.f64  Amount 0.5
//   prefix.coarse.f64   b through it with steps twice as large and a hold twice as long
//   prefix.same.f64     a with Amount in the middle (must be a itself);  prefix.own2.f64  Amount 2
//   prefix.owncoarse.f64  a with steps twice as large and a hold twice as long
// and, when plain sound was mixed in (the sound before the crusher can be had):
//   prefix.out.f64      a before the crusher;  prefix.own0.f64  the same from the app's own store (Amount 0);  prefix.own05.f64  Amount 0.5
//   prefix.ownfine.f64  a with steps half as large
// Nothing found: writes nothing.
//   drcrush find a.wav ...
// only what is found on each sound, one line of JSON each.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Crush.h"

static void writeRaw (const std::string& p, const dr::Stereo& s)
{
    std::ofstream f (p, std::ios::binary);
    f.write ((const char*) s.l.data(), (std::streamsize) (s.l.size() * sizeof (double))); f.write ((const char*) s.r.data(), (std::streamsize) (s.r.size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static dr::Stereo sound (const std::string& path, double& sr)
{
    dr::Stereo y; sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    return y;
}
static void say (const dr::crush::Found& F, double took)
{
    std::printf ("{\"took\": %.3f, \"found\": %s, \"kind\": %s, \"through\": %s, \"n\": %zu, \"from\": %ld, \"bits\": %.12g, \"P\": %.15g, \"c\": %.15g, \"step\": [%.17g, %.17g], \"offset\": [%.17g, %.17g], ",
                 took, F.found ? "true" : "false", ! F.found ? "null" : F.soft ? "\"soft\"" : "\"bare\"", F.through ? "true" : "false", F.n, F.from, F.bits, F.P, F.c, F.step[0], F.step[1], F.offset[0], F.offset[1]);
    std::printf ("\"mix\": %.12g, \"soft\": %.12g, \"half\": %.12g, \"inside\": %.12g, \"dense\": %.9g, \"lam\": %.9g, \"rhythm\": %.9g, \"aa_mass\": %.9g, \"aa_lam\": %.9g, \"grids\": [", F.mix, F.blend, F.half, F.inside, F.dense, F.lam, F.rhythm, F.aaMass, F.aaLam);
    for (int ch = 0; ch < 2; ++ch)
    {
        const auto& g = F.grids[ch];
        if (g.ok) std::printf ("%s{\"step\": %.17g, \"offset\": %.17g, \"share\": %.12g, \"levels\": %ld, \"bits\": %.12g}", ch ? ", " : "", g.step, g.offset, g.share, g.levels, g.bits);
        else std::printf ("%snull", ch ? ", " : "");
    }
    std::printf ("], \"hold\": {\"P\": %.15g, \"c\": %.15g, \"share\": %.12g, \"changes\": %ld}}\n", F.hold.P, F.hold.c, F.hold.share, F.hold.changes);
}
int main (int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock; namespace cr = dr::crush;
        if (argc >= 3 && std::string (argv[1]) == "find")      // only what is found: one line of JSON for each sound
        {
            for (int i = 2; i < argc; ++i) { double sr; const dr::Stereo a = sound (argv[i], sr); const auto t0 = clock::now(); const cr::Found F = cr::find (a, sr); say (F, std::chrono::duration<double> (clock::now() - t0).count()); std::fflush (stdout); }
            return 0;
        }
        if (argc < 4) { std::cout << "drcrush a.wav b.wav prefix | drcrush find a.wav ...\n"; return 1; }
        double sra, srb; const dr::Stereo a = sound (argv[1], sra), b = sound (argv[2], srb); const std::string pre = argv[3];
        const auto t0 = clock::now(); const cr::Found F = cr::find (a, sra); say (F, std::chrono::duration<double> (clock::now() - t0).count());
        if (! F.found) return 0;
        writeRaw (pre + ".onto.f64", cr::onto (b, F, 1.0)); writeRaw (pre + ".onto2.f64", cr::onto (b, F, 2.0)); writeRaw (pre + ".onto05.f64", cr::onto (b, F, 0.5));
        writeRaw (pre + ".coarse.f64", cr::onto (b, F, 1.0, 2.0, 2.0));
        writeRaw (pre + ".same.f64", cr::own (a, F, 1.0)); writeRaw (pre + ".own2.f64", cr::own (a, F, 2.0)); writeRaw (pre + ".owncoarse.f64", cr::own (a, F, 1.0, 2.0, 2.0));
        if (F.soft)
        {
            writeRaw (pre + ".out.f64", cr::softUndo (a, cr::softOf (F))); writeRaw (pre + ".own0.f64", cr::own (a, F, 0.0)); writeRaw (pre + ".own05.f64", cr::own (a, F, 0.5));
            writeRaw (pre + ".ownfine.f64", cr::own (a, F, 1.0, 0.5, 1.0));
        }
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drcrush: " << e.what() << "\n"; return 1; }
}
