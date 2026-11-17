// The verification step without the UI (48 kHz stereo WAV in), for comparison with the Python side
// (Tools/verify_parity.py) and for timing:
//   drverify ResourcesDir [leave=effect,effect...] sound.wav [sound.wav ...]
//   leave=...  effects the verifier is to leave alone, as an engine of its own would ask: in "final" they keep the
//              detectors' scores and "scores" does not have them
// prints one JSON line per sound:
//   n          samples
//   fx12, fx27 the two detectors' effects, in the order of the arrays below
//   win        per analysed 3 s window: start (sample), lg12 / lg27 (the CNNs' outputs), hand (the 15 hand numbers),
//              p12 / p27 (the deciders' scores)
//   blocks     per block the sound is judged in: start, len, win (which windows lie in it), echo (score, ms) and ev
//              (evidence(): the physical measurements by name)
//   scores     per effect the score of every judged block (blocks without a loud window are not judged)
//   before     the detectors' own scores, as they were shown up to 0.4.1
//   final      the verifier's scores: what the Sureness slider is compared with
//   ms         CPU time taken (of the process, so a busy machine does not count): inspect (both detectors), verify
//              (everything Verify.h does), evidence (evidence() alone, all blocks), echo (the echo check on the blocks;
//              a sound of one block takes it from the inspection)
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include "Wav.h"
#include "../Source/Core/Estimate.h"

template <typename V> static void arr (std::ostream& o, const V& v)
{
    o << "["; bool first = true;
    for (auto a : v) { o << (first ? "" : ","); if (std::isfinite ((double) a)) o << a; else o << "null"; first = false; }
    o << "]";
}
template <typename M> static void obj (std::ostream& o, const M& m)
{
    o << "{"; bool first = true;
    for (auto& [k, v] : m) { o << (first ? "" : ",") << "\"" << k << "\":"; if (std::isfinite ((double) v)) o << v; else o << "null"; first = false; }
    o << "}";
}
static std::vector<char> readAll (const std::string& p)
{
    std::ifstream f (p, std::ios::binary); if (! f) throw std::runtime_error ("cannot read " + p);
    return std::vector<char> ((std::istreambuf_iterator<char> (f)), {});
}
int main (int argc, char** argv)
{
    if (argc < 3) { std::cerr << "usage: drverify ResourcesDir [leave=effect,effect...] sound.wav [sound.wav ...]\n"; return 2; }
    try
    {
        const std::string dir = argv[1]; dr::Models M, M27; dr::Verifier V;
        { const auto b = readAll (dir + "/models.bin"); M.load (b.data(), b.size()); }
        { const auto b = readAll (dir + "/models27.bin"); M27.load (b.data(), b.size()); }
        { const auto b = readAll (dir + "/verify.bin"); V.load (b.data(), b.size()); }
        auto now = [] { return 1e3 * (double) std::clock() / (double) CLOCKS_PER_SEC; };
        auto& o = std::cout; o << std::setprecision (17); std::set<std::string> leave;
        for (int a = 2; a < argc; ++a)
        {
            if (std::string (argv[a]).rfind ("leave=", 0) == 0) { std::stringstream names (argv[a] + 6); std::string f; while (std::getline (names, f, ',')) if (! f.empty()) leave.insert (f); continue; }
            dr::Stereo y; double sr = 0; if (! dr::readWav (argv[a], y, sr)) { std::cerr << "cannot read " << argv[a] << "\n"; return 1; }
            const double t0 = now(); auto I = dr::inspect (M, y); dr::mergeNew (I, dr::inspect (M27, y, {}, &I));
            const double t1 = now(); const std::map<std::string, double> before = I.prob;
            std::map<std::string, dr::Vec> scores; dr::verify (I, y, V, leave, {}, &scores);
            const double t2 = now();
            o << "{\"file\":\"" << argv[a] << "\",\"n\":" << y.size() << ",\"fx12\":[";
            for (size_t i = 0; i < I.fx.size(); ++i) o << (i ? "," : "") << "\"" << I.fx[i] << "\"";
            o << "],\"fx27\":["; for (size_t i = 0; i < I.fx27.size(); ++i) o << (i ? "," : "") << "\"" << I.fx27[i] << "\"";
            o << "],\"win\":[";
            for (size_t r = 0; r < I.win.size(); ++r)
            {
                o << (r ? "," : "") << "{\"start\":" << I.win[r].start << std::setprecision (9) << ",\"lg12\":"; arr (o, I.win[r].logits); o << ",\"hand\":"; arr (o, I.win[r].hand);
                o << ",\"p12\":"; arr (o, I.win[r].p); o << ",\"lg27\":"; arr (o, I.win27[r].logits); o << ",\"p27\":"; arr (o, I.win27[r].p); o << "}" << std::setprecision (17);
            }
            o << "],\"blocks\":["; const auto bl = dr::verif::blocks (y.size()); const size_t W = (size_t) (3 * dr::SR); double msEv = 0, msEcho = 0; bool firstBlock = true;
            for (const auto& [b0, len] : bl)
            {
                const dr::Stereo part = bl.size() == 1 ? y : y.slice (b0, len);
                double e0 = now(); const auto ev = dr::evidence (part); msEv += now() - e0;
                std::pair<double, double> echo { I.echoScore, I.echoMs }; if (bl.size() > 1) { e0 = now(); echo = dr::hand::echoDetect (part); msEcho += now() - e0; }
                o << (firstBlock ? "" : ",") << "{\"start\":" << b0 << ",\"len\":" << len << ",\"win\":["; firstBlock = false; bool first = true;
                for (size_t r = 0; r < I.win.size(); ++r) { const size_t s = I.win[r].start; if (len <= W ? s == b0 : (s >= b0 && s + W <= b0 + len)) { o << (first ? "" : ",") << r; first = false; } }
                o << "],\"echo\":[" << echo.first << "," << echo.second << "],\"ev\":"; obj (o, ev); o << "}";
            }
            o << "],\"scores\":{"; bool first = true;
            for (auto& [f, v] : scores) { o << (first ? "" : ",") << "\"" << f << "\":"; arr (o, v); first = false; }
            std::map<std::string, double> fin; for (auto& f : V.families) fin[f.name] = I.prob[f.name];
            o << "},\"before\":"; obj (o, before); o << ",\"final\":"; obj (o, fin);
            o << ",\"ms\":{\"inspect\":" << t1 - t0 << ",\"verify\":" << t2 - t1 << ",\"evidence\":" << msEv << ",\"echo\":" << msEcho << "}}\n" << std::flush;
        }
    }
    catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
    return 0;
}
