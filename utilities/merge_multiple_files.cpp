#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "Not enough command line arguments: " << argc << " comand line arguments" << endl;
    cout << "Put in files <input 1> <input 2> ... <input n> <output> as arguments." << endl;
    return 1;
  }
  //n-2 input files
  int n_input_files = argc - 2;
  vector<ifstream> fins(n_input_files);
  for (int i = 1; i <= n_input_files; ++i) {
    fins[i-1].open(argv[i]);
  }
  ofstream fout(argv[argc - 1]);
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
        for (auto &tok : tokens) {
          sum += stold(tok);
        }
        sum /= tokens.size();
        fout << ',' << sum;
      }
      ++count;
    }
    fout << endl;
  }
}