#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 5) {
    cout << "Not enough command line arguments: " << argc << " comand line arguments" << endl;
    cout << "Put in files <input 1> <input 2> ... <input n> <weights> <output> as arguments." << endl;
    cout << "Weights is a file of space or newline separated weights" << endl;
    return 1;
  }
  //num inputs - 3 input files
  int n_input_files = argc - 3;
  vector<ifstream> fins(n_input_files);
  for (int i = 1; i <= n_input_files; ++i) {
    fins[i-1].open(argv[i]);
  }
  ifstream fweights(argv[argc-2]);
  ofstream fout(argv[argc - 1]);

  vector<long double> weights;
  long double w;
  long double total_weight = 0;
  for (int i = 0; i < fins.size(); ++i) {
    fweights >> w;
    weights.emplace_back(w);
    total_weight += w;
  }

  string header;
  for (auto &file : fins) {
    getline(file, header);
  }
  fout << header << endl;
  vector<string> lines(n_input_files);
  vector<string> tokens(n_input_files);
  long double sum;
  while (getline(fins[0], lines[0])) {
    vector<istringstream> sses(n_input_files);
    sses[0].str(lines[0]);
    for (int i = 1; i < fins.size(); ++i) {
      getline(fins[i], lines[i]);
      sses[i].str(lines[i]);
    }
    int count = 0;
    while (getline(sses[0], tokens[0], ',')) {
      for (int i = 1; i < fins.size(); ++i) {
        getline(sses[i], tokens[i], ',');
      }
      if (count == 0) {
        fout << tokens[0];
      } else {
        sum = 0;
        for (int i = 0; i < fins.size(); ++i) {
          sum += (stold(tokens[i]) * weights[i]);
        }
        sum /= total_weight;
        fout << ',' << sum;
      }
      ++count;
    }
    fout << endl;
  }
}