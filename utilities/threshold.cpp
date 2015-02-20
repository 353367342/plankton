#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
  if (argc != 4) {
    cout << "Usage is <input file> <output file> <threshold value>" << endl;
    return 1; 
  }
  ifstream fin(argv[1]);
  ofstream fout(argv[2]);
  long double threshold = stold(argv[3]);
  string header;
  getline(fin, header);
  fout << header << endl;
  string line, token;
  long double token_value;
  while (getline(fin, line)) {
    istringstream ss(line);
    int count = 0;
    while (getline(ss, token, ',')) {
      if (count == 0) {
        fout << token;
      } else {
        token_value = stold(token);
        if (token_value <= threshold) {
          fout << "," << 0;
        } else {
          fout << "," << token_value;
        }
      }
      ++count;
    }
    fout << endl;
  }

}