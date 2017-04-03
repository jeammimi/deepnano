#include <cstdio>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <algorithm>

using namespace std;

class string2 {
 public:
  string2() {}
  string2(const char*x) {
    x_[0] = 0;
    x_[1] = 0;
    x_[2] = 0;
    for (int i = 0; i < 2 && x[i]; i++) {
      x_[i] = x[i];
    }
  }

  string2 operator=(const char*x) {
    x_[0] = 0;
    x_[1] = 0;
    x_[2] = 0;
    for (int i = 0; i < 2 && x[i]; i++) {
      x_[i] = x[i];
    }
    return *this;
  }

  string2 operator=(const string2 x) {
    x_[0]= x.x_[0];
    x_[1]= x.x_[1];
    x_[2]= x.x_[2];
    return *this;
  }

  string2 operator=(const string& x) {
    x_[0] = 0;
    x_[1] = 0;
    x_[2] = 0;
    for (int i = 0; i < 2 && i < x.size(); i++) {
      x_[i] = x[i];
    }
    return *this;
  }

  char& operator[](int i) {
    return x_[i];
  }

  char x_[3];
};

int main() {
  char ss[100000];
  scanf("%s\n", ss);

  string ref(ss);
  int mapping[256];
  mapping['A'] = 0;
  mapping['C'] = 1;
  mapping['G'] = 2;
  mapping['T'] = 3;
  mapping['B'] = 4; //Modif
  int unknown = 5;
  int range = 4000;
  vector<vector<double>> probs;
  while (true) {
    double a, c, g, t, b, n;
    if (scanf("%lf %lf %lf %lf %lf %lf", &a, &c, &g, &t, &b, &n)<6) { //Modif
      break;
    }
    probs.push_back(vector<double>({log(a), log(c), log(g), log(t), log(b), log(n)})); // Modif
  }
  while (true) {
    vector<vector<double>> poses(probs.size()+1);
    vector<vector<string2>> prevs(probs.size()+1);
    for (int i = 0; i < poses.size(); i+=2) {
      poses[i] = vector<double>(ref.size()+1, -1e30);
      prevs[i] = vector<string2>(ref.size()+1, "");
    }
    poses[0][0] = 0;
    for (int i = 0; i < poses[0].size() && i < poses[0].size() - 500; i++) {
      poses[0][i] = 0;
    }
    int last_bp = 50;
    for (int i = 2; i <= probs.size(); i+=2) {
      for (int j = max(1, last_bp - range); j <= ref.size() && j <= last_bp + range; j++) {
        // NN
        if (i > 2) {
          double np = poses[i-2][j] + probs[i-2][unknown] + probs[i-1][unknown];
          if (np > poses[i][j]) {
            poses[i][j] = np;
            prevs[i][j] = "NN";
          }
        }
        // NX
        double np = poses[i-2][j-1] + probs[i-2][unknown] + probs[i-1][mapping[ref[j-1]]];
        if (np > poses[i][j]) {
          poses[i][j] = np;
          prevs[i][j] = string("N") + ref[j-1];
        }

        //XX
        if (j > 1) {
          double np = poses[i-2][j-2] + probs[i-2][mapping[ref[j-2]]] + probs[i-1][mapping[ref[j-1]]];
          if (np > poses[i][j]) {
            poses[i][j] = np;
            prevs[i][j] = ref.substr(j-2, 2);
          }
        }
      }
      int cur_bp = max(1, last_bp - range);
      for (int j = max(1, last_bp - range); j <= ref.size() && j <= last_bp + range; j++) {
        if (poses[i][j] > poses[i][cur_bp]) {
          cur_bp = j;
        }
      }
      last_bp = cur_bp;
    }
//    fprintf(stderr, "back size %d last_bp %d\n", poses.back().size(), last_bp);
    int best_pos = poses.back().size()-40;
    for (int i = min(500, (int)poses.back().size()-500); i < poses.back().size(); i++) {
      if (poses.back()[i] > poses.back()[best_pos]) {
        best_pos = i;
      }
    }
    /*int total = 0;
    int rel = 0;
    int better = 0;
    for (int i = 0; i < poses.size(); i += 2) {
      for (int j = 0; j < poses[i].size(); j++) {
        total += 1;
        if (poses[i][j] > -1e29) {
          rel += 1;
        }
        if (poses[i][j] > poses.back()[best_pos]) {
          better += 1;
        }
      }
    }
    fprintf(stderr, "total %d rel %d better %d\n", total, rel, better);*/
    if (poses.back()[best_pos] / probs.size() * 2 > -10) {
      fprintf(stderr, "best pos %d %lf %d\n", best_pos, poses.back()[best_pos] / probs.size() * 2, range);
      int ipos = poses.size()-1;
      int jpos = best_pos;

      vector<string2> out;
      while (ipos > 0) {
        auto back = prevs[ipos][jpos];
        out.push_back(back);
        if (back[0] == 'N' && back[1] == 'N') {
          ipos -= 2;
        }
        else if (back[0] == 'N' && back[1] != 'N') {
          ipos -= 2;
          jpos -= 1;
        }
        else if (back[0] != 'N' && back[1] != 'N') {
          ipos -= 2;
          jpos -= 2;
        }
      }
      reverse(out.begin(), out.end());
      for (auto &o: out) {
        printf("%s\n", o.x_);
      }
      fprintf(stderr, "start pos %d\n", jpos);
      return 0;
    }
    range *= 2;
  }
}
