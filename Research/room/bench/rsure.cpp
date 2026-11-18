// What the room engine says about each sound of a list, for a study of its sureness: one JSON line per sound.
//   build (from this folder): g++ -O2 -std=c++17 -pthread -o rsure rsure.cpp
//   rsure ResourcesDir list.txt [looks]      (list: one WAV path per line; a tab and a number after the path: only the first so many seconds;
//                                         looks: which looks besides the sound as it is, "end", "mono", "pieces" (a sound of more
//                                         than 9 s as the app judges it: in pieces of 6 s) or several with commas between; default "end")
// Per sound: the engine's answer on the sound as it is and with 3 s of silence after it (the app's two looks): found,
// the app's sureness, its evidence, and per band strength, decay time, evidence, width; plus how the model's answer is
// spread (shares of the live cells under -3, -6, -12, -20 dB; the room part's share of the energy per third of the bands).
#include <fstream>
#include <iostream>
#include <iterator>
#include "../../../Tests/Wav.h"
#include "../../../Source/Core/Estimate.h"
// The evidence of dr::roomEvidence stretch by stretch: stretches of `span` frames, each starting half a stretch after the one before
// and the last one ending with the sound (a sound no longer than one stretch: the whole sound, with the numbers
// roomEvidence gives). deep and bands as above, counted inside the stretch; live = the cells within 50 dB of the
// sound's loudest that the stretch holds.
struct RoomSpan { double deep = 0; int bands = 0; double live = 0; };
static std::vector<RoomSpan> roomSpans (const dr::room::Found& f, int span)
{
    std::vector<RoomSpan> out; const size_t n = f.gains.size(), NB = (size_t) dr::room::NB; const int T = (int) (n / NB);
    if (! f.found || n == 0 || span < 1 || f.picture.LE.size() != n) return out;
    const std::vector<unsigned char> moments = dr::room::roomNumbers (f.picture.LE, f.gains, T, true).moments; if (moments.size() != n) return out;
    const double top = f.picture.top(); const int hop = std::max (1, span / 2);
    for (int a = 0; ; a += hop)
    {
        const int from = std::max (0, std::min (a, T - span)), to = std::min (T, from + span);
        RoomSpan s; size_t live = 0, clear = 0; std::array<int, dr::room::NB> c {};
        for (size_t i = (size_t) from * NB; i < (size_t) to * NB; ++i)
        {
            c[i % NB] += moments[i];
            if (f.picture.LE[i] > top - 50.0) { ++live; clear += (double) f.gains[i] < -12.0 ? 1u : 0u; }
        }
        for (int v : c) s.bands += v >= 8;
        s.deep = live > 0 ? (double) clear / (double) live : 0.0; s.live = (double) live; out.push_back (s);
        if (to >= T) break;
    }
    return out;
}
static void look (const dr::Stereo& y, const dr::room::Net& net, const dr::room::Net* net2, const char* tag, bool pieces = false)
{
    const dr::room::Found f = dr::room::find (y, net, net2); const dr::RoomEvidence e = dr::roomEvidence (f);
    std::cout << "\"" << tag << "\": {\"found\": " << (f.found ? 1 : 0) << ", \"sure\": " << dr::roomSureness (f) << ", \"share\": " << e.share << ", \"deep\": " << e.deep << ", \"bands\": " << e.bands << ", \"moments\": " << e.moments
              << ", \"decay\": " << e.decay << ", \"strength\": " << e.strength << ", \"own\": " << e.own;
    const size_t n = f.gains.size(); const int NB = dr::room::NB, T = f.picture.frames;
    if (n > 0 && f.picture.LE.size() == n)
    {
        const double top = f.picture.top(); size_t live = 0, u3 = 0, u6 = 0, u12 = 0, u20 = 0; double room3[3] = { 0, 0, 0 }, all3[3] = { 0, 0, 0 };
        for (size_t i = 0; i < n; ++i)
        {
            if (! (f.picture.LE[i] > top - 50.0)) continue;
            const double g = f.gains[i], E = std::pow (10.0, (f.picture.LE[i] - top) / 10.0), G = std::pow (10.0, g / 10.0); ++live; u3 += g < -3; u6 += g < -6; u12 += g < -12; u20 += g < -20;
            const int b = (int) (i % (size_t) NB), k = b * 3 / NB; room3[k] += E * (1 - G); all3[k] += E;
        }
        const double L = (double) std::max<size_t> (live, 1);
        std::cout << ", \"frames\": " << T << ", \"live\": " << (double) live / (double) n << ", \"u3\": " << u3 / L << ", \"u6\": " << u6 / L << ", \"u12\": " << u12 / L << ", \"u20\": " << u20 / L;
        for (int k = 0; k < 3; ++k) std::cout << ", \"r" << k << "\": " << (all3[k] > 0 ? room3[k] / all3[k] : 0.0);
        // how the file begins and ends: the level of its first and last tenth of a second against its loudest moment (dB); near 0 = cut off while playing
        std::vector<double> e ((size_t) T, 0.0); double mx = 1e-30;
        for (int t = 0; t < T; ++t) { for (int b = 0; b < NB; ++b) e[(size_t) t] += std::pow (10.0, f.picture.LE[(size_t) t * (size_t) NB + (size_t) b] / 10.0); mx = std::max (mx, e[(size_t) t]); }
        auto lev = [&] (int a, int b) { double s = 0; int c = 0; for (int t = std::max (a, 0); t < std::min (b, T); ++t) { s += e[(size_t) t]; ++c; } return 10.0 * std::log10 (std::max (c > 0 ? s / c : 0.0, 1e-30) / mx); };
        std::cout << ", \"startDb\": " << lev (4, 14) << ", \"endDb\": " << lev (T - 15, T - 5);
        // the evidence stretch by stretch (stretches of 9.1 s, 6.1 s and 4.5 s): [deep, bands, live cells] each
        for (int span : { 852, 568, 426 })
        {
            std::cout << ", \"spans" << span << "\": ["; bool first = true;
            for (const RoomSpan& sp : roomSpans (f, span)) { std::cout << (first ? "" : ",") << "[" << sp.deep << "," << sp.bands << "," << sp.live << "]"; first = false; }
            std::cout << "]";
        }
    }
    // a sound of more than 9 s as the app judges it (Estimate.h, roomInPieces): the verdict, the verdict of the pieces
    // alone (the app shows nothing when the whole sound has no room) and per piece [start in s, sureness, weight]
    if (pieces && T > dr::roomWholeFrames)
    {
        std::vector<dr::RoomPiece> each; const double v = dr::roomInPieces (y, net, net2, {}, &each);
        std::cout << ", \"verdict\": " << (f.found ? v : dr::roomSureness (f)) << ", \"inpieces\": " << v << ", \"pieces\": [";
        for (size_t i = 0; i < each.size(); ++i) std::cout << (i ? "," : "") << "[" << (double) each[i].from / 48000.0 << "," << each[i].sure << "," << each[i].weight << "]";
        std::cout << "]";
    }
    for (int q = 0; q < 4; ++q)
    {
        const dr::room::Band& B = q == 0 ? f.kdb : q == 1 ? f.t60 : q == 2 ? f.ev : f.width; std::cout << ", \"" << (q == 0 ? "kdb" : q == 1 ? "t60" : q == 2 ? "ev" : "width") << "\": [";
        for (int b = 0; b < NB; ++b) std::cout << (b ? "," : "") << (f.found ? (double) B[(size_t) b] : 0.0);
        std::cout << "]";
    }
    std::cout << "}";
}
int main (int argc, char** argv)
{
    if (argc < 3) return 1;
    const std::string dir = argv[1]; dr::room::Net net, net2;
    { std::ifstream f (dir + "/room.bin", std::ios::binary); const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); net.load (b.data(), b.size()); }
    { std::ifstream f (dir + "/room2.bin", std::ios::binary); if (f) { const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); net2.load (b.data(), b.size()); } }
    std::ifstream L (argv[2]); std::string p; const std::string looks = argc > 3 ? argv[3] : "end";
    while (std::getline (L, p))
    {
        if (p.empty()) continue;
        double cut = 0; const auto tab = p.find ('\t'); if (tab != std::string::npos) { cut = std::stod (p.substr (tab + 1)); p = p.substr (0, tab); }
        dr::Stereo y; double sr = 0; if (! dr::readWav (p, y, sr)) { std::cerr << "cannot read " << p << "\n"; continue; }
        if (cut > 0 && (size_t) (cut * 48000.0) < y.size()) { y.l.resize ((size_t) (cut * 48000.0)); y.r.resize ((size_t) (cut * 48000.0)); }
        std::cout << "{\"path\": \"" << p << "\", \"cut\": " << cut << ", \"sec\": " << (double) y.size() / 48000.0 << ", "; look (y, net, net2.ready() ? &net2 : nullptr, "as", looks.find ("pieces") != std::string::npos);
        if (looks.find ("end") != std::string::npos)        // 3 s of silence after the sound, as the app's end look had it up to 0.4.4
        {
            dr::Stereo z = y; const size_t n = y.size() + (size_t) (3.0 * dr::SR); z.l.resize (n, 0.0); z.r.resize (n, 0.0);
            std::cout << ", "; look (z, net, net2.ready() ? &net2 : nullptr, "end");
        }
        if (looks.find ("mono") != std::string::npos)       // left and right summed: what is left of the room when the two sides are one
        {
            dr::Stereo z = y; for (size_t i = 0; i < z.size(); ++i) z.l[i] = z.r[i] = 0.5 * (y.l[i] + y.r[i]);
            std::cout << ", "; look (z, net, net2.ready() ? &net2 : nullptr, "mono");
        }
        std::cout << "}" << std::endl;
    }
    return 0;
}
