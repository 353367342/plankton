#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

int main(int argc, char **argv) {
  if (argc != 4) {
    cout << "Put in files <input1> <input2> <output> as arguments." << endl;
  }
  ifstream fin0(argv[1]);
  ifstream fin1(argv[2]);
  ofstream fout(argv[3]);
  string header;
  getline(fin0, header);
  getline(fin1, header);
  fout << header << endl;

  string line0, line1, id;
  long double a, b;
  string token0, token1;

  while (getline(fin0, line0)) {
    getline(fin1, line1);
    istringstream ss0(line0);
    istringstream ss1(line1);
    int count = 0;
    while (getline(ss0, token0, ',')) {
      getline(ss1, token1, ',');
      if (count == 0) {
        fout << token0;
      } else {
        a = stold(token0);
        b = stold(token1);
        fout << ',' << (a+b) / 2;
      }
      ++count;
    }
    fout << endl;
  }
}