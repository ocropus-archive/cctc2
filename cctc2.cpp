#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <future>

#ifndef MAXEXP
#define MAXEXP 30
#endif

template<typename T, typename ... Args>
void print() {
  std::cerr << "\n";
}

template<typename T, typename ... Args>
void print(T first) {
  std::cerr << first << "\n";
}

template<typename T, typename ... Args>
void print(T first, Args ... args) {
  std::cerr << first << " ";
  print(args ...);
}

using torch::Tensor;

Tensor d_sigmoid(Tensor z) {
  auto s = torch::sigmoid(z);
  return (1-s)*s;
}

void square(Tensor a) {
  for(int i=0; i<a.size(0); i++)
    for(int j=0; j<a.size(1); j++)
      a[i][j] = a[i][j] * a[i][j];
}

void make_one(Tensor a) {
  a.resize_({1, 1});
  a.fill_(1);
}


inline int rows(Tensor m) {
  return m.size(0);
}

inline int cols(Tensor m) {
  if (m.dim()!=2) abort();
  return m.size(1);
}

inline float limexp(float x) {
  if (x < -MAXEXP) return exp(-MAXEXP);
  if (x > MAXEXP) return exp(MAXEXP);
  return exp(x);
}

inline float log_add(float x, float y) {
  if (fabs(x - y) > 10) return fmax(x, y);
  return log(exp(x - y) + 1) + y;
}

inline float log_mul(float x, float y) {
  return x + y;
}

inline float &aref1(const Tensor &a, int i) {
  return a.accessor<float, 1>()[i];
}

inline float &aref2(const Tensor &a, int i, int j) {
  return a.accessor<float, 2>()[i][j];
}

bool check_rownorm(Tensor a) {
  for(int i=0; i<a.size(0); i++) {
    double total = 0.0;
    for(int j=0; j<a.size(1); j++) {
      double value = aref2(a, i, j);
      if (value<0) return false;
      if (value>1) return false;
      total += value;
    }
    if (abs(total-1.0) > 1e-4) return false;
  }
  return true;
}

static Tensor forward_algorithm(Tensor lmatch, double skip = -5) {
  print("forward");
  int n = rows(lmatch), m = cols(lmatch);
  Tensor lr = torch::zeros({n, m});
  Tensor v = torch::zeros(m);
  Tensor w = torch::zeros(m);
  for (int j = 0; j < m; j++) aref1(v, j) = skip * j;
  for (int i = 0; i < n; i++) {
    aref1(w, 0) = skip * i;
    for (int j = 1; j < m; j++) aref1(w, j) = aref1(v, j - 1);
    for (int j = 0; j < m; j++) {
      float same = log_mul( aref1(v, j), aref2(lmatch, i, j));
      float next = log_mul( aref1(w, j), aref2(lmatch, i, j));
      aref1(v, j) = log_add(same, next);
    }
    for (int j = 0; j < m; j++) aref2(lr, i, j) = aref1(v, j);
  }
  return lr;
}


static Tensor forwardbackward(Tensor lmatch) {
  print("forwardbackward");
  int n = rows(lmatch), m = cols(lmatch);
  Tensor lr = forward_algorithm(lmatch);
  Tensor rlmatch = torch::zeros({n, m});
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      aref2(rlmatch, i, j) = aref2(lmatch, n - i - 1, m - j - 1);
  Tensor rrl = forward_algorithm(rlmatch);
  Tensor rl = torch::zeros({n, m});
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      aref2(rl, i, j) = aref2(rrl, n - i - 1, m - j - 1);
  return lr + rl;
}

Tensor ctc_align_targets(Tensor outputs, Tensor targets) {
  print("align");
  assert(cols(targets) == cols(outputs));
  assert(rows(targets) <= rows(outputs));
  assert(check_rownorm(outputs));
  assert(check_rownorm(targets));
  double lo = 1e-6;
  int n1 = rows(outputs);
  int n2 = rows(targets);
  int nc = cols(targets);

  // compute log probability of state matches
  print("align1");
  Tensor lmatch = torch::zeros({n1, n2});
  print("align2");
  for (int t1 = 0; t1 < n1; t1++) {
    Tensor out = torch::zeros(nc);
    for (int i = 0; i < nc; i++) aref1(out, i) = fmax(lo, aref2(outputs, t1, i));
    out = out / out.sum();
    for (int t2 = 0; t2 < n2; t2++) {
      double total = 0.0;
      for (int k = 0; k < nc; k++) total += aref1(out, k) * aref2(targets, t2, k);
      aref2(lmatch, t1, t2) = log(total);
    }
  }
  // compute unnormalized forward backward algorithm
  Tensor both = forwardbackward(lmatch);

  // compute normalized state probabilities
  Tensor epath = both - both.max();
  for(int i=0; i<epath.size(0); i++) 
    for(int j=0; j<epath.size(1); j++) 
      aref2(epath, i, j) = limexp( aref2(epath, i, j));
  for (int j = 0; j < n2; j++) {
    double total = 0.0;
    for (int i = 0; i < rows(epath); i++) total += aref2(epath, i, j);
    total = fmax(1e-9, total);
    for (int i = 0; i < rows(epath); i++) aref2(epath, i, j) /= total;
  }

  // compute posterior probabilities for each class and normalize
  Tensor aligned = torch::zeros({n1, nc});
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < nc; j++) {
      double total = 0.0;
      for (int k = 0; k < n2; k++) {
        double value = aref2(epath, i, k) * aref2(targets, k, j);
        total += value;
      }
      aref2(aligned, i, j) = total;
    }
  }
  for (int i = 0; i < n1; i++) {
    double total = 0.0;
    for (int j = 0; j < nc; j++) total += aref2(aligned, i, j);
    total = fmax(total, 1e-9);
    for (int j = 0; j < nc; j++) aref2(aligned, i, j) /= total;
  }
  assert(check_rownorm(aligned));
  return aligned;
}

Tensor ctc_align_targets_batch(Tensor outputs, Tensor targets) {
  assert(outputs.dim()==3);
  assert(targets.dim()==3);
  int b = outputs.size(0), n = outputs.size(1), m = outputs.size(2);
  Tensor posteriors = torch::zeros({b, n, m});
  if(getenv("CTC_NOTHREAD") && atoi(getenv("CTC_NOTHREAD"))) {
    for(int i=0; i<b; i++) {
      Tensor o = outputs.select(0, i);
      Tensor t = targets.select(0, i);
      posteriors.select(0, i) = ctc_align_targets(o, t);
    }
  } else {
    int bs = posteriors.size(0);
    std::vector<std::future<int> > results(bs);
    for(int i=0; i<b; i++) {
      results[i] = std::async(
			      std::launch::async,
			      [i, &posteriors, &outputs, &targets]() {
				Tensor o = outputs.select(0, i);
				Tensor t = targets.select(0, i);
				posteriors.select(0, i) = ctc_align_targets(o, t);
				return 1;
			      });
    }
    for(int i=0; i<b; i++) {
      results[i].wait();
    }
  }
  return posteriors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d_sigmoid", &d_sigmoid, "d_sigmoid");
  m.def("square", &square, "square");
  m.def("make_one", &make_one, "make_one");
  m.def("check_rownorm", &check_rownorm, "check_rownorm");
  m.def("forward_algorithm", &forward_algorithm, "forward_algorithm");
  m.def("forwardbackward", &forwardbackward, "forwardbackward");
  m.def("ctc_align_targets", &ctc_align_targets, "ctc_align_targets");
  m.def("ctc_align_targets_batch", &ctc_align_targets_batch, "ctc_align_targets_batch");
}

