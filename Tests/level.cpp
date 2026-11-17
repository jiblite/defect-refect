// The level-movement engine without the UI (48 kHz WAV in), for comparison with the Python prototype
// (Tools/level_parity.py) and for scoring it:
//   drlevel measure a.wav [least=0]                 one line of JSON: what was found and everything measured
//   drlevel out a.wav out [s=-1] [least=0]          a with its movement taken out (s = -1) or scaled (s between -1 and 2)
//   drlevel onto a.wav b.wav out [s=1] [least=0]    a's movement put onto b
// and as the app uses it (a sound of any length, measured piece by piece; one line of JSON about the pieces, the
// stretches that were asked whether the movement is there, and the stretches that have it):
//   drlevel pieces a.wav
//   drlevel own a.wav out [amount=0] [speed=1]      a with its own movement as the faders ask (0 = taken out, 1 = untouched, 2 = twice)
//   drlevel put a.wav b.wav out [amount=1] [speed=1]    a's movement on b
//   drlevel both a.wav b.wav outA outB              one analysis: a without its movement, and a's movement on b
//   drlevel at a.wav from length hz                 a stretch of a (seconds) measured at a given speed: the measures, what the first filter says
// Options of these: carry= (the score from which a piece carries the movement), price= and lean= (where the movement
// is: what a change between 'there' and 'not there' costs, and how far a piece that carries it leans towards 'there').
// `out` ending in .f64 is written as raw float64 (left, right, left, ...), anything else as a float32 WAV.
// least: the lowest score that still counts as found (0 = as the prototype; the app goes down to -2 and lets the
// Sureness slider decide).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Wav.h"
#include "../Source/Core/Level.h"

static bool writeOut (const std::string& p, const dr::Stereo& y)
{
    std::ofstream f (p, std::ios::binary); const uint32_t n = (uint32_t) y.size();
    if (p.size() > 4 && p.substr (p.size() - 4) == ".f64")
    {
        std::vector<double> v ((size_t) n * 2); for (uint32_t i = 0; i < n; ++i) { v[(size_t) i * 2] = y.l[i]; v[(size_t) i * 2 + 1] = y.r[i]; }
        f.write ((const char*) v.data(), (std::streamsize) (v.size() * 8)); return (bool) f;
    }
    const uint32_t data = n * 2 * 4; auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); }; auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
    f.write ("RIFF", 4); w32 (36 + data); f.write ("WAVEfmt ", 8); w32 (16); w16 (3); w16 (2); w32 (48000); w32 (48000 * 8); w16 (8); w16 (32); f.write ("data", 4); w32 (data);
    std::vector<float> v ((size_t) n * 2); for (uint32_t i = 0; i < n; ++i) { v[(size_t) i * 2] = (float) y.l[i]; v[(size_t) i * 2 + 1] = (float) y.r[i]; }
    f.write ((const char*) v.data(), (std::streamsize) (v.size() * 4));
    return (bool) f;
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    if (std::abs (sr - 48000.0) > 0.5) throw std::runtime_error (path + " is not at 48 kHz");
    return y;
}
static void num (const char* k, double v, bool last = false) { if (std::isfinite (v)) std::printf ("\"%s\": %.17g%s", k, v, last ? "" : ", "); else std::printf ("\"%s\": null%s", k, last ? "" : ", "); }
static void arr (const char* k, const dr::Vec& v, bool last = false)
{
    std::printf ("\"%s\": [", k); for (size_t i = 0; i < v.size(); ++i) { if (std::isfinite (v[i])) std::printf ("%s%.17g", i ? ", " : "", v[i]); else std::printf ("%snull", i ? ", " : ""); }
    std::printf ("]%s", last ? "" : ", ");
}
static void ints (const char* k, const std::vector<int>& v, bool last = false) { std::printf ("\"%s\": [", k); for (size_t i = 0; i < v.size(); ++i) std::printf ("%s%d", i ? ", " : "", v[i]); std::printf ("]%s", last ? "" : ", "); }
static void tog (const dr::level::Tog& m)
{
    const auto& a = m.a; std::printf ("{"); num ("P", m.P); std::printf ("\"K\": %d, \"start\": %d, \"cycles\": %d, \"bands\": %d, \"cyc_n\": %d, ", m.K, m.start, m.cycles, a.bands, a.cyc_n); ints ("skip", m.skip); arr ("g", m.g);
    num ("r2", m.r2); num ("depth", m.depth); num ("skipfrac", m.skipfrac); num ("up", m.up); num ("steep", m.steep); num ("fund", m.fund); num ("rise", m.rise); num ("fall", m.fall); num ("climb", m.climb); num ("flat", m.flat);
    num ("share", m.share); num ("excess", m.excess); num ("slope_med", a.slope_med); num ("slope_lo", a.slope_lo); num ("slope_hi", a.slope_hi); num ("slope_min", a.slope_min); num ("slope_max", a.slope_max);
    num ("corr_med", a.corr_med); num ("corr_lo", a.corr_lo); num ("corr_min", a.corr_min); num ("cyc_med", a.cyc_med); num ("cyc_lo", a.cyc_lo); num ("cyc_hi", a.cyc_hi); num ("halves", a.halves); num ("r2c", a.r2c);
    num ("silent", a.silent); num ("dis_med", a.dis_med); num ("dis_hi", a.dis_hi); num ("score", m.score, true); std::printf ("}");
}
static double opt (int argc, char** argv, int from, const char* key, double dflt)
{
    const size_t n = std::strlen (key);
    for (int i = from; i < argc; ++i) if (std::strncmp (argv[i], key, n) == 0 && argv[i][n] == '=') return std::atof (argv[i] + n + 1);
    return dflt;
}

int main (int argc, char** argv)
{
    try
    {
        const std::string mode = argc > 1 ? argv[1] : "";
        using clock = std::chrono::steady_clock; const auto t0 = clock::now(); auto secs = [&] { return std::chrono::duration<double> (clock::now() - t0).count(); };
        if (mode == "measure" && argc >= 3)
        {
            const dr::Stereo a = sound (argv[2]); const auto t1 = clock::now(); const dr::level::Found F = dr::level::find (a, {}, opt (argc, argv, 3, "least", 0.0)); const double took = std::chrono::duration<double> (clock::now() - t1).count();
            std::printf ("{\"secs\": %.6f, \"took\": %.3f, \"found\": %s, \"kind\": \"%s\", ", (double) a.size() / dr::SR, took, F.found ? "true" : "false", F.found ? (F.against ? "against" : "together") : "");
            num ("score", F.score); num ("sure", F.sure); num ("period", F.period); num ("rate", F.rate); num ("depth", F.depth); num ("trust", F.trust); num ("law", F.law); num ("lawfit", F.lawfit);
            std::printf ("\"K\": %d, \"start\": %d, ", F.K, F.start); ints ("skip", F.skip); ints ("open", F.open); arr ("profL", F.prof[0]); arr ("profR", F.prof[1]);
            std::printf ("\"lane\": %zu, \"edges\": [", F.lane[0].size());
            for (size_t i = 0; i < F.edges.size(); ++i) { const auto& e = F.edges[i]; std::printf ("%s{", i ? ", " : ""); num ("off", e.off); std::printf ("\"half\": %d, \"k\": %d, ", e.half, e.k); arr ("u", e.u, true); std::printf ("}"); }
            const auto& p = F.pan; std::printf ("], \"pan\": {\"ok\": %s, \"slow\": %s, \"K\": %d, \"bands\": %d, ", p.ok ? "true" : "false", p.slow ? "true" : "false", p.K, p.bands);
            num ("P", p.P); num ("centre", p.centre); num ("swing", p.swing); num ("share", p.share); num ("adj", p.adj); num ("resid", p.resid); num ("cycles", p.cycles); num ("slope_med", p.slope_med); num ("slope_lo", p.slope_lo);
            num ("slope_hi", p.slope_hi); num ("known", p.known); num ("fund", p.fund); num ("steep", p.steep); arr ("prof", p.prof); num ("score", p.score, true);
            std::printf ("}, \"cands\": ["); for (size_t i = 0; i < F.cands.size(); ++i) { if (i) std::printf (", "); tog (F.cands[i]); }
            std::printf ("]}\n"); return 0;
        }
        if (mode == "out" && argc >= 4)
        {
            const dr::Stereo a = sound (argv[2]); const dr::level::Found F = dr::level::find (a, {}, opt (argc, argv, 4, "least", 0.0)); const double ta = secs();
            if (! writeOut (argv[3], dr::level::apply (a, F, opt (argc, argv, 4, "s", -1.0)))) throw std::runtime_error ("cannot write the result");
            std::fprintf (stderr, "found %d (%s, %.3f Hz, %.1f dB deep, sure %.3f), analysis %.2f s, render %.2f s\n", (int) F.found, F.against ? "against" : "together", F.rate, F.depth, F.sure, ta, secs() - ta); return 0;
        }
        if (mode == "onto" && argc >= 5)
        {
            const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); const dr::level::Found F = dr::level::find (a, {}, opt (argc, argv, 5, "least", 0.0)); const double ta = secs();
            if (! writeOut (argv[4], dr::level::apply (b, F, opt (argc, argv, 5, "s", 1.0)))) throw std::runtime_error ("cannot write the result");
            std::fprintf (stderr, "found %d (%s, %.3f Hz, %.1f dB deep, sure %.3f), analysis %.2f s, render %.2f s\n", (int) F.found, F.against ? "against" : "together", F.rate, F.depth, F.sure, ta, secs() - ta); return 0;
        }
        auto pieces = [&] (const dr::level::Movement& M, double took)
        {
            std::printf ("{\"took\": %.3f, \"found\": %s, \"kind\": \"%s\", \"steps\": %s, \"sounding\": %d, \"carrying\": %d, \"lead\": %d, \"wide\": %s, ", took, M.found ? "true" : "false", M.found ? (M.against ? "against" : "together") : "", M.steps ? "true" : "false", M.sounding, M.carrying, M.lead, M.wide ? "true" : "false"); std::printf ("\"steady\": %s, \"knots\": %zu, ", M.clock.steady ? "true" : "false", M.clock.t.size());
            num ("score", M.score); num ("sure", M.sure); num ("rate", M.rate); num ("depth", M.depth); std::printf ("\"pieces\": [");
            for (size_t i = 0; i < M.pieces.size(); ++i)
            {
                const auto& p = M.pieces[i]; std::printf ("%s{\"a\": %zu, \"b\": %zu, \"sounds\": %s, \"use\": %s, \"guided\": %s, \"found\": %s, \"against\": %s, \"edges\": %zu, \"skip\": %zu, ", i ? ", " : "", p.a, p.b, p.sounds ? "true" : "false", p.use ? "true" : "false",
                                                           p.guided ? "true" : "false", p.f.found ? "true" : "false", p.f.against ? "true" : "false", p.f.edges.size(), p.f.skip.size());
                std::printf ("\"clocked\": %s, ", p.clocked ? "true" : "false"); num ("there", p.there);
                num ("own", p.own); num ("score", p.f.score); num ("rate", p.f.rate); num ("depth", p.f.depth); num ("trust", p.f.trust, true); std::printf ("}");
            }
            std::printf ("], \"windows\": ["); for (size_t i = 0; i < M.windows.size(); ++i) { const auto& w = M.windows[i]; std::printf ("%s[%zu, %zu, %.4f, %.4f, %d, %d]", i ? ", " : "", w.a, w.b, w.u, w.w, (int) w.own, (int) w.on); }
            std::printf ("], \"spans\": ["); for (size_t i = 0; i < M.spans.size(); ++i) std::printf ("%s[%zu, %zu]", i ? ", " : "", M.spans[i].a, M.spans[i].b);
            std::printf ("]}\n");
        };
        if (mode == "at" && argc >= 6)           // a stretch of a (from, length in s) measured at a given speed (Hz): the measures and what the first filter says
        {
            const dr::Stereo a = sound (argv[2]); const size_t from = (size_t) (std::atof (argv[3]) * dr::SR), len = (size_t) (std::atof (argv[4]) * dr::SR);
            const dr::Stereo x = a.slice (from, std::min (len, a.size() - std::min (from, a.size()))); const dr::level::Levels E = dr::level::levels (x); dr::level::Tog m;
            const bool ok = dr::level::togetherAt (E, dr::SR / dr::level::HOP / std::atof (argv[5]), m); const auto& g = m.a;
            std::printf ("measured %d  P %.2f frames (%.3f Hz)  score %.2f  first filter %d  struck %d\n", (int) ok, m.P, dr::SR / dr::level::HOP / std::max (m.P, 1e-9), m.score, (int) dr::level::firstFilter (m), (int) dr::level::struck (m));
            std::printf ("  depth %.2f (>= 3)  corr_lo %.3f (>= 0.6)  corr_med %.3f  cyc_n %d (>= 4)  halves %.3f (>= 0.8)  slope lo %.2f med %.2f hi %.2f (spread <= 0.5, med within 0.5 of 1)  share %.3f (>= %.2f)  excess %.3f  cyc lo %.2f hi %.2f (spread <= %.0f)\n",
                         m.depth, g.corr_lo, g.corr_med, g.cyc_n, g.halves, g.slope_lo, g.slope_med, g.slope_hi, m.share, dr::level::SHARE_MIN, m.excess, g.cyc_lo, g.cyc_hi, dr::level::CYC_SPREAD);
            std::printf ("  fall %.3f climb %.3f flat %.3f  fall*P %.1f (struck: fall >= 0.15, climb <= %.2f fall, fall*P >= 20)  steep %.2f fund %.2f r2 %.3f bands %d silent %.2f skipfrac %.2f\n", m.fall, m.climb, m.flat, m.fall * m.P, dr::level::STRUCK, m.steep, m.fund, m.r2, g.bands, g.silent, m.skipfrac);
            return 0;
        }
        if (mode == "pieces" && argc >= 3)       // a whole sound, however long, as the app measures it
        {
            const dr::Stereo a = sound (argv[2]); const dr::level::Movement M = dr::level::analyse (a, {}, opt (argc, argv, 3, "carry", dr::level::CARRY), opt (argc, argv, 3, "price", dr::level::SWITCH), opt (argc, argv, 3, "lean", dr::level::LEAN)); pieces (M, secs()); return 0;
        }
        if (mode == "both" && argc >= 6)         // one analysis of a: a without its movement, and a's movement on b
        {
            const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); const dr::level::Movement M = dr::level::analyse (a, {}, opt (argc, argv, 6, "carry", dr::level::CARRY), opt (argc, argv, 6, "price", dr::level::SWITCH), opt (argc, argv, 6, "lean", dr::level::LEAN)); const double ta = secs(); dr::Vec gl, gr;
            dr::level::ownDb (M, a.size(), 0.0, 1.0, gl, gr); if (! writeOut (argv[4], dr::level::withDb (a, gl, gr))) throw std::runtime_error ("cannot write the result");
            dr::level::ontoDb (M, b.size(), 1.0, 1.0, gl, gr); if (! writeOut (argv[5], dr::level::withDb (b, gl, gr))) throw std::runtime_error ("cannot write the result");
            pieces (M, ta); return 0;
        }
        if (mode == "own" && argc >= 4)          // the sound with its own movement as the faders ask (default amount=0: taken out)
        {
            const dr::Stereo a = sound (argv[2]); const dr::level::Movement M = dr::level::analyse (a, {}, opt (argc, argv, 4, "carry", dr::level::CARRY), opt (argc, argv, 4, "price", dr::level::SWITCH), opt (argc, argv, 4, "lean", dr::level::LEAN)); const double ta = secs(); dr::Vec gl, gr;
            dr::level::ownDb (M, a.size(), opt (argc, argv, 4, "amount", 0.0), opt (argc, argv, 4, "speed", 1.0), gl, gr);
            if (! writeOut (argv[3], dr::level::withDb (a, gl, gr))) throw std::runtime_error ("cannot write the result");
            pieces (M, ta); return 0;
        }
        if (mode == "put" && argc >= 5)          // a's movement on b (default amount=1: as found)
        {
            const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); const dr::level::Movement M = dr::level::analyse (a, {}, opt (argc, argv, 5, "carry", dr::level::CARRY), opt (argc, argv, 5, "price", dr::level::SWITCH), opt (argc, argv, 5, "lean", dr::level::LEAN)); const double ta = secs(); dr::Vec gl, gr;
            dr::level::ontoDb (M, b.size(), opt (argc, argv, 5, "amount", 1.0), opt (argc, argv, 5, "speed", 1.0), gl, gr);
            if (! writeOut (argv[4], dr::level::withDb (b, gl, gr))) throw std::runtime_error ("cannot write the result");
            pieces (M, ta); return 0;
        }
        std::cout << "drlevel measure a.wav [least=0] | out a.wav out [s=-1] [least=0] | onto a.wav b.wav out [s=1] [least=0] | pieces a.wav | own a.wav out [amount=0] [speed=1] | put a.wav b.wav out [amount=1] [speed=1] | both a.wav b.wav outA outB\n"; return 1;
    }
    catch (const std::exception& e) { std::cerr << "drlevel: " << e.what() << "\n"; return 1; }
}
