// Small PHDST/ZEBRA bank lister for DETRAW smoke tests.
//
// It walks the current event's RAW/DST/TANAGRA top banks and prints a compact
// tree of ZEBRA bank names, with raw-like Rxxx banks highlighted.

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"
#include "phdst.hpp"
#include "phdst_analysis.hpp"

namespace ph = phdst;

namespace
{
struct Options
{
    std::filesystem::path pdlinput;
    int maxEvents = 1;
    int maxDepth = 8;
    int maxLinks = 64;
    bool allBanks = false;
    bool rawOnly = true;
};

std::string trim_bank_name(std::string name)
{
    while (!name.empty() && name.back() == ' ')
    {
        name.pop_back();
    }
    return name;
}

std::string hollerith4(int word)
{
    char bytes[4];
    std::memcpy(bytes, &word, sizeof(bytes));
    std::string little(bytes, bytes + 4);
    std::string big{bytes[3], bytes[2], bytes[1], bytes[0]};

    auto printableScore = [](const std::string &s) {
        int score = 0;
        for (unsigned char c : s)
        {
            if (std::isupper(c) || std::isdigit(c) || c == ' ')
            {
                ++score;
            }
        }
        return score;
    };

    const std::string &chosen = (printableScore(little) >= printableScore(big)) ? little : big;
    return chosen;
}

std::string bank_name(int l)
{
    if (l <= 0)
    {
        return "";
    }
    return hollerith4(ph::IQ(l - 4));
}

int bank_nlinks(int l)
{
    return l > 0 ? ph::IQ(l - 2) : 0;
}

int bank_ndata(int l)
{
    return l > 0 ? ph::IQ(l - 1) : 0;
}

bool is_raw_like(const std::string &name)
{
    const std::string trimmed = trim_bank_name(name);
    if (trimmed == "RAW")
    {
        return true;
    }
    if (trimmed.size() >= 2 && trimmed[0] == 'R')
    {
        return true;
    }
    return false;
}

std::string path_with_link(const std::string &parentPath, const std::string &name, int linkIndex)
{
    std::ostringstream out;
    out << parentPath << "/-" << linkIndex << ":" << trim_bank_name(name);
    return out.str();
}

class RawBankLister : public phdst::Analysis
{
public:
    static RawBankLister *getInstance()
    {
        if (instance_ == nullptr)
        {
            instance_ = new RawBankLister();
        }
        return static_cast<RawBankLister *>(instance_);
    }

    void setOptions(const Options &options)
    {
        options_ = options;
        setMaxEventsToProcess(options_.maxEvents);
    }

    void user00() override
    {
        phdst::Analysis::user00();
        std::cout << "RawBankLister: maxEvents=" << options_.maxEvents
                  << " maxDepth=" << options_.maxDepth
                  << " maxLinks=" << options_.maxLinks
                  << " mode=" << (options_.rawOnly ? "raw-like banks" : "all banks")
                  << std::endl;
    }

    void user02() override
    {
        phdst::Analysis::user02();
        ++eventIndex_;
        seen_.clear();
        rawBankNames_.clear();

        std::cout << "\n=== event " << eventIndex_
                  << " run=" << ph::IIIRUN
                  << " evt=" << ph::IIIEVT
                  << " record=" << ph::PHRTY()
                  << " ===" << std::endl;
        std::cout << "tops: LRTOP=" << ph::LRTOP
                  << " LDTOP=" << ph::LDTOP
                  << " LTTOP=" << ph::LTTOP
                  << " LITOP=" << ph::LITOP
                  << " LRTINT=" << ph::LRTINT
                  << std::endl;

        walkTop("RAW_TOP", ph::LRTOP);
        walkTop("DST_TOP", ph::LDTOP);
        walkTop("TAN_TOP", ph::LTTOP);
        walkTop("INT_TOP", ph::LITOP);
        walkTop("RAW_INT_TOP", ph::LRTINT);

        if (rawBankNames_.empty())
        {
            std::cout << "summary: no RAW/Rxxx-like banks seen from PHDST-visible tops" << std::endl;
        }
        else
        {
            std::cout << "summary: RAW/Rxxx-like bank names:";
            for (const auto &name : rawBankNames_)
            {
                std::cout << ' ' << name;
            }
            std::cout << std::endl;
        }
    }

private:
    RawBankLister() = default;

    void walkTop(const std::string &label, int top)
    {
        if (top <= 0)
        {
            return;
        }
        std::cout << "-- " << label << " --" << std::endl;
        walkChain(top, label, 0);
    }

    void walkChain(int l, const std::string &parentPath, int depth)
    {
        for (int current = l; current > 0; current = ph::LQ(current))
        {
            if (seen_.count(current) != 0)
            {
                printLine(current, parentPath + "/chain", depth, "cycle");
                return;
            }
            seen_.insert(current);

            const std::string name = bank_name(current);
            const bool rawLike = is_raw_like(name);
            if (rawLike)
            {
                rawBankNames_.insert(trim_bank_name(name));
            }

            if (!options_.rawOnly || rawLike)
            {
                printLine(current, parentPath, depth, rawLike ? "raw" : "");
            }

            if (depth >= options_.maxDepth)
            {
                continue;
            }

            const int nlinks = std::clamp(bank_nlinks(current), 0, options_.maxLinks);
            for (int i = 1; i <= nlinks; ++i)
            {
                const int child = ph::LQ(current - i);
                if (child <= 0)
                {
                    continue;
                }
                walkChain(child, path_with_link(parentPath, name, i), depth + 1);
            }
        }
    }

    void printLine(int l, const std::string &path, int depth, const std::string &tag) const
    {
        const std::string name = bank_name(l);
        std::cout << std::string(static_cast<std::size_t>(depth) * 2, ' ')
                  << trim_bank_name(name)
                  << " addr=" << l
                  << " nl=" << bank_nlinks(l)
                  << " nd=" << bank_ndata(l);
        if (!tag.empty())
        {
            std::cout << " [" << tag << "]";
        }
        std::cout << " path=" << path << std::endl;
    }

    Options options_;
    int eventIndex_ = 0;
    std::set<int> seen_;
    std::set<std::string> rawBankNames_;
};

void configure_parser(argparse::ArgumentParser &program)
{
    program.add_argument("-P", "--pdlinput")
        .metavar("FILE")
        .help("Path to a PDLINPUT file, or directly to a .dst/.sdst file")
        .required();

    program.add_argument("-m", "--max-events")
        .metavar("N")
        .default_value(1)
        .scan<'i', int>();

    program.add_argument("--max-depth")
        .metavar("N")
        .default_value(8)
        .scan<'i', int>();

    program.add_argument("--max-links")
        .metavar("N")
        .default_value(64)
        .scan<'i', int>();

    program.add_argument("--all")
        .help("Print all visible banks instead of only RAW/Rxxx-like banks")
        .default_value(false)
        .implicit_value(true);
}

bool looks_like_pdl(const std::filesystem::path &path)
{
    const auto filename = path.filename().string();
    if (filename == "PDLINPUT" || path.extension() == ".pdl")
    {
        return true;
    }

    std::ifstream in(path);
    std::string firstLine;
    if (std::getline(in, firstLine))
    {
        return firstLine.find("FILE") != std::string::npos ||
               firstLine.find("FAT") != std::string::npos ||
               firstLine.find("VSN") != std::string::npos;
    }
    return false;
}

int create_pdlinput(const std::filesystem::path &input)
{
    std::filesystem::remove("PDLINPUT");
    if (!std::filesystem::exists(input))
    {
        std::cerr << "ERROR: file " << input << " does not exist" << std::endl;
        return 1;
    }

    if (looks_like_pdl(input))
    {
        std::filesystem::create_symlink(input, "PDLINPUT");
        return 0;
    }

    std::ofstream out("PDLINPUT");
    if (!out.is_open())
    {
        std::cerr << "ERROR: cannot write PDLINPUT" << std::endl;
        return 1;
    }
    out << "FILE = " << input.string() << '\n';
    return 0;
}
} // namespace

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("delphi-raw-bank-lister", "0.0");
    configure_parser(program);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    Options options;
    options.pdlinput = program.get<std::string>("--pdlinput");
    options.maxEvents = program.get<int>("--max-events");
    options.maxDepth = program.get<int>("--max-depth");
    options.maxLinks = program.get<int>("--max-links");
    options.allBanks = program.get<bool>("--all");
    options.rawOnly = !options.allBanks;

    if (int rc = create_pdlinput(options.pdlinput))
    {
        return rc;
    }

    auto *lister = RawBankLister::getInstance();
    lister->setOptions(options);
    return lister->run(" ");
}
