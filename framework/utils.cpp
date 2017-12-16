//wuhaibing, 2017-12-16

#include "utils.h"

using namespace tiny_dnn;

void parse_conf(const std::string conf_file, Conf &conf) {
  std::ifstream fin(conf_file);
  if (!fin) {
    throw nn_error("failed to open file: "+conf_file);
  }
  std::string line;
  while (std::getline(fin, line)) {
    line.erase(0, line.find_first_not_of(' '));
    line.erase(line.find_last_not_of(' ')+1, line.size());
    if (line.size() <= 0) {
      continue;
    }
    if (line[0] == '#') {
      continue;
    }
    //std::cout << line << std::endl;
    size_t found = line.find_first_of(":");
    if (found == std::string::npos) {
      throw nn_error("invalid config format");
    }
    std::string conf_key = line.substr(0, found);
    std::string conf_val = line.substr(found+1, line.size()-found);
    conf_key.erase(conf_key.find_last_not_of(' ')+1, line.size());
    conf_val.erase(0, conf_val.find_first_not_of(' '));
    //std::cout << conf_key << "->" << conf_val << std::endl;
    conf[conf_key] = conf_val;
  }
  if (conf.find("train_file") == conf.end()) {
    throw nn_error("invalid conf: item[train_file] not found");
  }
  if (conf.find("test_file") == conf.end()) {
    throw nn_error("invalid conf: item[test_file] not found");
  }
  if (conf.find("fea_dim") == conf.end()) {
    throw nn_error("invalid conf: item[fea_dim] not found");
  }
  if (conf.find("net") == conf.end()) {
    throw nn_error("invalid conf: item[net] not found");
  }
  if (conf.find("optimizer") == conf.end()) {
    throw nn_error("invalid conf: item[optimizer] not found");
  }
  if (conf.find("learn_rate") == conf.end()) {
    throw nn_error("invalid conf: item[learn_rate] not found");
  }
  if (conf.find("momentum") == conf.end()) {
    throw nn_error("invalid conf: item[momentum] not found");
  }
  if (conf.find("epoch") == conf.end()) {
    throw nn_error("invalid conf: item[epochs] not found");
  }
  if (conf.find("batch_size") == conf.end()) {
    throw nn_error("invalid conf: item[batch_size] not found");
  }
}

void _parse_input(std::string input_file,
                 std::vector<vec_t> &data_x,
                 std::vector<vec_t> &data_y,
                 size_t dim_input) {
  std::ifstream fin(input_file);
  if (!fin) {
    throw nn_error("failed to open file: "+input_file);
  }
  std::string line;
  while (std::getline(fin, line)) {
    line.erase(0, line.find_first_not_of(' '));
    line.erase(line.find_last_not_of(' ')+1, line.size());
    if (line.size() <= 0) {
      continue;
    }
    std::stringstream ss(line);
    float_t target;
    float_t fea;
    vec_t feas, targets;
    try {
      ss >> target;
      targets.push_back(target);
      while (ss >> fea) {
        feas.push_back(fea);
      }
    }
    catch (const nn_error& e) {
      throw nn_error("failed to read label and feas");
    }
    if (feas.size() != dim_input) {
      throw nn_error("invalid data, slot num != fea_dim");
    }

    data_x.push_back(feas);
    data_y.push_back(targets);
  }
}

void parse_input(Conf conf,
                std::vector<vec_t> &train_x,
                std::vector<vec_t> &train_y,
                std::vector<vec_t> &test_x,
                std::vector<vec_t> &test_y) {
  std::string train_file = conf["train_file"];
  std::string test_file = conf["test_file"];
  std::string input_dim = conf["fea_dim"];

  size_t dim_input;
  try {
    std::stringstream ss(input_dim);
    ss >> dim_input;
  }
  catch (const nn_error& e) {
    throw nn_error("invalid fea_dim");
  }

  _parse_input(train_file, train_x, train_y, dim_input);
  if (test_file != "XX") {
    _parse_input(test_file, test_x, test_y, dim_input);
  }
}

void set_architecture(Conf conf, network<sequential> &net) {
    std::string net_arch_file = conf["net"];
    net.load(net_arch_file, content_type::model, file_format::json);
}

template <typename Opt>
void set_optimizer(Conf &conf, Opt &opt) {
  std::string optimizer = conf["optimizer"];
  std::string learn_rate = conf["learn_rate"];
  std::string mu = conf["momentum"];

  float_t lr = std::stof(learn_rate);
  float_t mum = std::stof(mu);

  if (optimizer != "momentum" &&
      optimizer != "adagrad" &&
      optimizer != "RMSprop") {
    throw nn_error("invalid optimizer: "+optimizer);
  }

  opt.alpha = lr;
  opt.mu = mum;
}

void train_net(Conf &conf,
               network<sequential> &net,
               std::vector<vec_t> &train_x,
               std::vector<vec_t> &train_y,
               std::vector<vec_t> &test_x,
               std::vector<vec_t> &test_y) {
  int epoch = std::stoi(conf["epoch"]);
  int batch_size = std::stoi(conf["batch_size"]);

  progress_display disp(train_x.size());
  timer t;

  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << t.elapsed() << "s elapsed." << std::endl;

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //shuffle (train_x.begin(), train_x.end(), std::default_random_engine(seed));
    float_t loss = net.get_loss<mse>(train_x, train_y);
    std::cout << "loss: " << loss << std::endl;

    if (test_x.size() > 0) {
      std::vector<vec_t> pred = net.test(test_x);
      float_t auc = calc_auc(test_y, pred);
      float_t acc = calc_acc(test_y, pred);
      std::cout << "auc " << auc << " " << "acc " << acc << std::endl;
    }

    disp.restart(train_x.size());
    t.restart();
  };

  auto on_enumerate_data = [&]() { ++disp; };

  if (conf["optimizer"]=="momentum") {
    momentum mu_opt;
    set_optimizer<momentum>(conf, mu_opt);
    net.fit<mse>(mu_opt,train_x,train_y,batch_size,epoch,
                   on_enumerate_data, on_enumerate_epoch);
  }
  else if (conf["optimizer"]=="adagrad") {
    adagrad ag_opt;
    set_optimizer<adagrad>(conf, ag_opt);
    net.fit<mse>(ag_opt,train_x,train_y,batch_size,epoch,
                   on_enumerate_data, on_enumerate_epoch);
  }
  else {
    RMSprop rp_opt;
    set_optimizer<RMSprop>(conf, rp_opt);
    net.fit<mse>(rp_opt,train_x,train_y,batch_size,epoch,
                   on_enumerate_data, on_enumerate_epoch);
  }
}

bool _comp(vec_t a, vec_t b) {
  return (a[1]>b[1]);
}

float_t calc_auc(std::vector<vec_t> &y, std::vector<vec_t> &pred) {
    vec_t yy, pp;
    for (auto a : y) yy.push_back(a[0]);
    for (auto b : pred) pp.push_back(b[0]);

    size_t n = pp.size();
    std::vector<vec_t> yp;
    for (size_t i=0; i<yy.size(); ++i) {
        vec_t elem;
        elem.push_back(yy[i]);
        elem.push_back(pp[i]);
        yp.push_back(elem);
    }
    std::sort(yp.begin(), yp.end(), _comp);
    float_t np = 0.0, nn = 0.0;

    for (auto& x : yy) {
        if (x > 0) np += 1.0;
        else nn += 1.0;
    }
    float_t s = 0.0;
    for (size_t j=0; j<yp.size(); ++j) {
        if (yp[j][0] > 0) s += (n-j);
    }
    return (s-np*(np+1)*0.5)/(np*nn);
}

float_t calc_acc(std::vector<vec_t> &labels, std::vector<vec_t> &preds) {
    vec_t yy, pp;
    for (auto a : labels) yy.push_back(a[0]);
    for (auto b : preds) pp.push_back(b[0]);

    float_t cnt = 0;
    for (size_t i=0; i<yy.size(); ++i) {
        if (pp[i] > 0.5 && yy[i] > 0.5) {
            cnt += 1;
        }
        if (pp[i] < 0.5 && yy[i] < 0.5) {
            cnt += 1;
        }
    }
    float_t acc = cnt / yy.size();
    return acc;
}
