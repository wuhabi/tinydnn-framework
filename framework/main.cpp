// wuhaibing, 2017-12-16

#include "tiny_dnn/tiny_dnn.h"
#include "utils.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        throw nn_error("pls specify config file");
    }
  
    std::string conf_file = argv[1];
    Conf conf;
    parse_conf(conf_file, conf);

    std::vector<vec_t> train_x, test_x;
    std::vector<vec_t> train_y, test_y;
    parse_input(conf, train_x, train_y, test_x, test_y);

    network<sequential> net;
    set_architecture(conf, net);

    train_net(conf, net, train_x, train_y, test_x, test_y);

    return 0;
}

