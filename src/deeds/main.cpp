#include "deeds/deedsBCV.h"

#include "argparse/argparse.hpp"

#include <filesystem>
#include <string>
#include <vector>

auto createArgParser() {
    argparse::ArgumentParser parser("deedsBCV", "1.0", argparse::default_arguments::help);
    parser.add_argument("-F", "--fixed")
        .help("Fixed image filename")
        .default_value(std::string("fixed.nii.gz"));
    parser.add_argument("-M", "--moving")
        .help("Moving image filename")
        .default_value(std::string("moving.nii.gz"));
    parser.add_argument("-a").help("Regularisation parameter alpha").default_value(1.6f).implicit_value(float{});
    parser.add_argument("-l").help("Number of levels").default_value(3).implicit_value(int{});
    parser.add_argument("-G")
        .help("Grid spacing for each level")
        .nargs(argparse::nargs_pattern::at_least_one)
        .default_value(std::vector<int>{8, 6, 4})
        .scan<'d', int>();
    parser.add_argument("-L")
        .help("Maximum search radius per level")
        .nargs(argparse::nargs_pattern::at_least_one)
        .default_value(std::vector<int>{8, 6, 4})
        .scan<'d', int>();
    parser.add_argument("-Q")
        .help("Quantisation of search step size")
        .nargs(argparse::nargs_pattern::at_least_one)
        .default_value(std::vector<int>{5, 2, 1})
        .scan<'d', int>();
    return parser;
}

int main(int argc, char *argv[]) {
    auto argparser = createArgParser();
    try {
        argparser.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << argparser;
        std::exit(1);
    }

    auto params = DeedsBCV::Parameters{};
    auto tmp = argparser.get<std::string>("--fixed");
    params.fixed_image = argparser.get<std::string>("--fixed");
    params.moving_image = argparser.get<std::string>("--moving");

    if (!std::filesystem::exists(params.fixed_image) || !std::filesystem::exists(params.moving_image)) {
        std::cerr << "Non-existent image files: " << params.fixed_image << " " << params.moving_image << std::endl;
    }

    auto reg = DeedsBCV(params);
    reg.execute();

    return 0;
}