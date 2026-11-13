// The echo engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/echo_parity.py)
// and for scoring it:
//   drecho measure a.wav                      one line of JSON: per channel every lag that stood out and everything measured
//                                             about the most echo-like ones (with the rule's answer), what was found, how sure
//   drecho out a.wav out.wav [fader=value ..]     a with its echoes as the faders ask (default amount=0: taken out)
//   drecho onto a.wav b.wav out.wav [fader=value ..]    a's echoes on b (default amount=1: as found)
//   drecho ir a.wav prefix [fader=value ..]       the filter that puts a's echoes on another sound: prefix.L.f64, prefix.R.f64 (raw float64)
// Faders: amount, time, feedback, bright (1 = as found; time is a factor).
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
#include "../Source/Core/Echo.h"

static bool writeWav (const std::string& p, const dr::Stereo& y)       // float32, level untouched
{
    std::ofstream f (p, std::ios::binary); const uint32_t n = (uint32_t) y.size(), data = n * 2 * 4;
    auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); }; auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
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
static dr::echo::Shape shape (int argc, char** argv, int from, double amount)
{
    dr::echo::Shape s; s.amount = amount;
    for (int i = from; i < argc; ++i)
    {
        const std::string a = argv[i]; const size_t e = a.find ('='); if (e == std::string::npos) throw std::runtime_error ("not a fader: " + a);
        const std::string k = a.substr (0, e); const double v = std::atof (a.c_str() + e + 1);
        if (k == "amount") s.amount = v; else if (k == "time") s.time = v; else if (k == "feedback") s.feedback = v; else if (k == "bright") s.bright = v; else throw std::runtime_error ("not a fader: " + k);
    }
    return s;
}
static void num (const char* k, double v, bool last = false) { if (std::isfinite (v)) std::printf ("\"%s\": %.17g%s", k, v, last ? "" : ", "); else std::printf ("\"%s\": null%s", k, last ? "" : ", "); }
static void record (const dr::echo::Record& r)
{
    std::printf ("{\"q\": %d, ", r.q); num ("ms", r.ms); num ("score", r.score); num ("a", r.a); num ("h", r.h); num ("a2", r.a2); num ("added", r.added); num ("removed", r.removed); num ("even", r.even);
    std::printf ("\"bands\": %d, \"others\": %d, \"npk\": %d, ", r.bands, r.others, r.npk); num ("secs", r.secs); num ("pshare", r.pshare); num ("pmed", r.pmed); num ("piqr", r.piqr); num ("pup", r.pup);
    std::printf ("\"pn\": %d, ", r.pn); num ("pinfo", r.pinfo); num ("peff", r.peff); num ("p", r.p, true); std::printf ("}");
}

int main (int argc, char** argv)
{
    try
    {
        const std::string mode = argc > 1 ? argv[1] : "";
        using clock = std::chrono::steady_clock; const auto t0 = clock::now(); auto secs = [&] { return std::chrono::duration<double> (clock::now() - t0).count(); };
        if (mode == "measure" && argc >= 3)
        {
            const dr::Stereo a = sound (argv[2]); const auto t1 = clock::now(); const dr::echo::Found f = dr::echo::analyse (a); const double took = std::chrono::duration<double> (clock::now() - t1).count();
            std::printf ("{\"secs\": %.6f, \"start\": %ld, \"length\": %ld, \"took\": %.3f, \"sure\": %.6f, \"p\": %.6f, \"ch\": [", (double) a.size() / dr::SR, f.start, f.length, took, f.sure, f.p);
            for (int c = 0; c < 2; ++c)
            {
                std::printf ("%s{\"cand\": [", c ? ", " : "");
                for (size_t i = 0; i < f.cand[c].size(); ++i) { const auto& k = f.cand[c][i]; std::printf ("%s{\"q\": %d, ", i ? ", " : "", k.q); num ("score", k.score); num ("h", k.h); num ("a", k.a); std::printf ("\"rah\": %s}", k.rah ? "true" : "false"); }
                std::printf ("], \"recs\": ["); for (size_t i = 0; i < f.seen[c].size(); ++i) { if (i) std::printf (", "); record (f.seen[c][i]); }
                std::printf ("], \"found\": "); if (f.ch[c].found) { std::printf ("{\"q\": %d, \"p\": %.6f, \"phi\": %.4f, \"level\": [", f.ch[c].rec.q, f.ch[c].rec.p, f.ch[c].phi); for (size_t i = 0; i < f.ch[c].level.size(); ++i) std::printf ("%s%.2f", i ? ", " : "", f.ch[c].level[i]); std::printf ("]}"); } else std::printf ("null");
                std::printf ("}");
            }
            std::printf ("]}\n"); return 0;
        }
        if (mode == "out" && argc >= 4)
        {
            const dr::Stereo a = sound (argv[2]); const dr::echo::Found f = dr::echo::analyse (a); const double ta = secs();
            const dr::Stereo y = dr::echo::render (a, f, shape (argc, argv, 4, 0.0), true); if (! writeWav (argv[3], y)) throw std::runtime_error ("cannot write the result");
            std::fprintf (stderr, "found %d %d (sure %.3f), analysis %.2f s, render %.2f s\n", (int) f.ch[0].found, (int) f.ch[1].found, f.sure, ta, secs() - ta); return 0;
        }
        if (mode == "onto" && argc >= 5)
        {
            const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); const dr::echo::Found f = dr::echo::analyse (a); const double ta = secs();
            const dr::Stereo y = dr::echo::render (b, f, shape (argc, argv, 5, 1.0), false); if (! writeWav (argv[4], y)) throw std::runtime_error ("cannot write the result");
            std::fprintf (stderr, "found %d %d (sure %.3f), analysis %.2f s, render %.2f s\n", (int) f.ch[0].found, (int) f.ch[1].found, f.sure, ta, secs() - ta); return 0;
        }
        if (mode == "ir" && argc >= 4)
        {
            const dr::Stereo a = sound (argv[2]); const dr::echo::Found f = dr::echo::analyse (a); const dr::echo::Shape sh = shape (argc, argv, 4, 1.0); dr::echo::Fft F;
            for (int c = 0; c < 2; ++c)
            {
                const dr::Vec h = f.ch[c].found ? dr::echo::response (F, f.ch[c], sh, false) : dr::Vec { 1.0 };
                std::ofstream o (std::string (argv[3]) + (c ? ".R.f64" : ".L.f64"), std::ios::binary); o.write ((const char*) h.data(), (std::streamsize) (h.size() * sizeof (double)));
                std::fprintf (stderr, "%s: %zu samples\n", c ? "right" : "left", h.size());
            }
            return 0;
        }
        std::cout << "drecho measure a.wav | out a.wav out.wav [fader=value ..] | onto a.wav b.wav out.wav [fader=value ..] | ir a.wav prefix [fader=value ..]\n"; return 1;
    }
    catch (const std::exception& e) { std::cerr << "drecho: " << e.what() << "\n"; return 1; }
}
