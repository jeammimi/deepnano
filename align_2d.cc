#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

struct Prob {
  double p[4];
  double n;
};

Prob LoadProb(int parity) {
  Prob p;
  scanf("%lf %lf %lf %lf %lf", &p.p[0], &p.p[1], &p.p[2], &p.p[3], &p.n);
  for (int i = 0; i < 4; i++) {
    p.p[i] = log(p.p[i]);
  }
  p.n = log(p.n); 
  return p;
}

Prob LoadProbC(int parity) {
  Prob p;
  scanf("%lf %lf %lf %lf %lf", &p.p[3], &p.p[2], &p.p[1], &p.p[0], &p.n);
  for (int i = 0; i < 4; i++) {
    p.p[i] = log(p.p[i]);
  }
  p.n = log(p.n);
  return p;
}

int main() {
  vector<Prob> temp, comp;
  int tc, cc;
  scanf("%d", &tc);
  for (int i = 0; i < tc; i++) {
    temp.push_back(LoadProb(i%2));
  }
  for (int i = 0; i < tc; i+=2) {
    temp[i].n -= log(0.9) / 2;
  }
  scanf("%d", &cc);
  for (int i = 0; i < cc; i++) {
    comp.push_back(LoadProbC(i%2));
  }
  for (int i = 0; i < cc; i+=2) {
    comp[i].n -= log(0.9) / 2;
  }
  reverse(comp.begin(), comp.end());
  vector<vector<double>> probs(temp.size()+1, vector<double>(comp.size()+1));
  // 0 - 3 ACGT
  // 4 - Ncomp
  // 5 - Ntemp
  vector<vector<int>> prevs(temp.size()+1, vector<int>(comp.size()+1));

  double minf = -1000000000000.0;
  double pen = -0.5;
  probs[0][0] = 0;
  for (int i = 0; i < probs.size(); i++) {
    for (int j = 0; j < probs[i].size(); j++) {
      if (i == 0 && j == 0) continue;
      probs[i][j] = 0;
//      if (i == 0 && (j < 2000 && j < probs[i].size() / 4)) probs[i][j] = 0;
//      if (j == 0 && (i < 2000 && i < probs.size() / 4)) probs[i][j] = 0;
      prevs[i][j] = 6;
      if (j > 0) {
        double np = probs[i][j-1] + comp[j-1].n - pen;
        if (np > probs[i][j]) {
          prevs[i][j] = 4;
          probs[i][j] = np;
        }
      }
      if (i > 0) {
        double np = probs[i-1][j] + temp[i-1].n - pen;
        if (np > probs[i][j]) {
          prevs[i][j] = 5;
          probs[i][j] = np;
        }
      }
      if (i > 0 && j > 0) {
        for (int k = 0; k < 4; k++) {
          double np = probs[i-1][j-1] + (temp[i-1].p[k] + comp[j-1].p[k]) - 2*pen;
          if (np > probs[i][j]) {
            prevs[i][j] = k;
            probs[i][j] = np;
          }
        }
      }
    }
  }
  fprintf(stderr, "%lf\n", probs.back().back());

  char alph[] = "ACGT";
  string seq;
  int ipos = temp.size();
  int jpos = comp.size();
/*  int margin = min(2000, (int)temp.size() / 4);
  for (int i = temp.size(); i >= temp.size() - margin && i >= 0; i--) {
    if (probs[i][comp.size()] > probs[ipos][jpos]) {
      ipos = i;
      jpos = comp.size();
    }
  }
  margin = min(2000, (int)comp.size() / 4);
  for (int j = comp.size(); j >= comp.size() - margin && j >= 0; j--) {
    if (probs[temp.size()][j] > probs[ipos][jpos]) {
      ipos = temp.size();
      jpos = j;
    }
  }*/
  for (int i = 0; i < temp.size(); i++) {
    for (int j = 0; j < comp.size(); j++) {
      if (probs[i][j] > probs[ipos][jpos]) {
        ipos = i;
        jpos = j;
      }
    }
  }

  vector<pair<int, int>> trace;
  while (ipos > 0 && jpos > 0) {
    if (prevs[ipos][jpos] == 6) {
      break;
    }
    trace.push_back(make_pair(ipos, jpos));
    if (prevs[ipos][jpos] == 4) {
      jpos--;
    } else if (prevs[ipos][jpos] == 5) {
      ipos--;
    } else {
      seq += alph[prevs[ipos][jpos]];
      ipos--;
      jpos--;
    }
  }
  reverse(trace.begin(), trace.end());
  fprintf(stderr, "%d\n", seq.size());
  reverse(seq.begin(), seq.end());
  printf("%s\n", seq.c_str());
  
  int last_temp = -47;
  int last_comp = -47;
  for (int i = 10; i + 10 < trace.size(); i++) {
    auto t = trace[i];
    int temp_al = -1;
    int comp_al = -1;
    if (t.first != last_temp && t.first % 2 == 0) {
      temp_al = (t.first - 1) / 2;
    }
    if (t.second != last_comp && t.second % 2 == 1) {
      comp_al = comp.size() / 2 - 2 - t.second / 2;
    }
    if (temp_al != -1 || comp_al != -1) {
      printf("%d %d\n", temp_al, comp_al);
    }
    last_temp = t.first;
    last_comp = t.second;
  }
}
