// Command-line driver for delphi-raw-nanoaod.
//
// Patterned on delphi-nanoaod/src/delphi-nanoaod.cpp but minus the YAML
// configuration (no SKELANA flags to twiddle) and minus the --mc/--data
// flags (the raw banks are the raw banks — MC truth would come from a
// separate PA.SIM extra-module read, left for a later milestone).

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "argparse/argparse.hpp"
#include "raw_nanoaod_writer.hpp"

static void configure_parser(argparse::ArgumentParser &program)
{
    auto &group = program.add_mutually_exclusive_group(true);

    group.add_argument("-N", "--nickname")
        .metavar("NICKNAME")
        .help("Fatmen nickname of the sample");

    group.add_argument("-P", "--pdlinput")
        .metavar("FILE")
        .help("Path to the pdl input file (or direct SDST path)");

    program.add_argument("-O", "--output")
        .metavar("FILE")
        .help("Output ROOT file name")
        .required();

    program.add_argument("-m", "--max-events")
        .metavar("N")
        .help("Maximum number of events to process")
        .scan<'i', int>();

    program.add_argument("--no-raw-bank-scan")
        .help("Disable the generic RAW/Rxxx ZEBRA bank manifest and payload scan")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--raw-bank-all")
        .help("Scan every visible bank, not only RAW/Rxxx-like banks")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--raw-bank-max-depth")
        .metavar("N")
        .help("Maximum ZEBRA tree depth for the generic raw bank scan")
        .default_value(8)
        .scan<'i', int>();

    program.add_argument("--raw-bank-max-links")
        .metavar("N")
        .help("Maximum child links visited per bank during the raw bank scan")
        .default_value(64)
        .scan<'i', int>();

    program.add_argument("--raw-bank-max-words")
        .metavar("N")
        .help("Maximum payload words saved per raw bank; use -1 to save all words, 0 for metadata only")
        .default_value(64)
        .scan<'i', int>();
}

static bool looks_like_pdl(const std::filesystem::path &path)
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

static int create_pdlinput(const argparse::ArgumentParser &program)
{
    std::filesystem::remove("PDLINPUT");

    if (auto nickname = program.present("--nickname"))
    {
        std::ofstream pdlinput("PDLINPUT");
        if (!pdlinput.is_open())
        {
            std::cerr << "ERROR: cannot open PDLINPUT for writing" << std::endl;
            return 1;
        }
        pdlinput << "FAT = " << nickname.value() << std::endl;
    }
    else
    {
        auto pdlinput = std::filesystem::path(program.get("--pdlinput"));
        if (!std::filesystem::exists(pdlinput))
        {
            std::cerr << "ERROR: file " << pdlinput << " does not exist" << std::endl;
            return 1;
        }
        if (looks_like_pdl(pdlinput))
        {
            std::filesystem::create_symlink(pdlinput, "PDLINPUT");
        }
        else
        {
            std::ofstream out("PDLINPUT");
            if (!out.is_open())
            {
                std::cerr << "ERROR: cannot open PDLINPUT for writing" << std::endl;
                return 1;
            }
            out << "FILE = " << pdlinput.string() << '\n';
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("delphi-raw-nanoaod", "0.0");
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

    auto *writer = RawNanoAODWriter::getInstance();
    writer->configureRawBankScan(
        !program.get<bool>("--no-raw-bank-scan"),
        program.get<int>("--raw-bank-max-depth"),
        program.get<int>("--raw-bank-max-links"),
        program.get<int>("--raw-bank-max-words"),
        program.get<bool>("--raw-bank-all"));

    if (int rc = create_pdlinput(program))
    {
        return rc;
    }

    if (auto maxEvents = program.present<int>("--max-events"))
    {
        writer->setMaxEventsToProcess(maxEvents.value());
    }

    auto output = program.get<std::string>("--output");
    writer->setOutput(output);

    return writer->run(" ");
}
