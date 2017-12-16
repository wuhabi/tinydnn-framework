//wuhaibing, 2017-12-16

#include "tiny_dnn/tiny_dnn.h"
#include <iostream>
#include <unordered_map>
#include <fstream>

using namespace tiny_dnn;

typedef std::unordered_map<std::string,std::string> Conf;

void parse_conf(const std::string conf_file, Conf &conf);

void parse_input(Conf conf,
                std::vector<vec_t> &train_x,
                std::vector<vec_t> &train_y,
                std::vector<vec_t> &test_x,
                std::vector<vec_t> &test_y);

void set_architecture(Conf conf, network<sequential> &net);

template <typename Opt>
void set_optimizer(Conf &conf, Opt &opt);

void train_net(Conf &conf,
               network<sequential> &net,
               std::vector<vec_t> &train_x,
               std::vector<vec_t> &train_y,
               std::vector<vec_t> &test_x,
               std::vector<vec_t> &test_y);

float_t calc_auc(std::vector<vec_t> &y,
                 std::vector<vec_t> &pred);
