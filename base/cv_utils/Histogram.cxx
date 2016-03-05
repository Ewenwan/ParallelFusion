#include "cv_utils.h"

#include <cmath>

using namespace std;

namespace cv_utils
{
  template<typename T> Histogram<T>::Histogram(const int NUM_HISTOS, const T MIN_VALUE, const T MAX_VALUE, const vector<T> &values) : NUM_GRAMS_(NUM_HISTOS), MIN_VALUE_(MIN_VALUE), MAX_VALUE_(MAX_VALUE), num_values_(values.size(), histos_(vector<int>(NUM_HISTOS, 0)))
  {
    for (typename vector<T>::const_iterator value_it = values.begin(); value_it != values.end(); value_it++)
      if (*value_it >= MIN_VALUE_ && *value_it <= MAX_VALUE)
	histos_[calcHistoIndex(*value_it)]++;
  }
  
  template<typename T> double Histogram<T>::getEntropy()
  {
    double entropy = 0;
    for (vector<int>::const_iterator histo_it = histos_.begin(); histo_it != histos_.end(); histo_it++)
      entropy += -1.0 * *histo_it / num_values_ * log(1.0 * *histo_it / num_values_);
    return entropy;
  }
  
  template<typename T> double Histogram<T>::getProbability(const T &value)
  {
    if (value < MIN_VALUE_ || value > MAX_VALUE_)
      return 0;
    return histos_[calcHistoIndex(value)];
  }
  
  template<typename T> double Histogram<T>::calcHistoIndex(const T &value)
  {
    return min(max(static_cast<int>(1.0 * (value - MIN_VALUE_) / (MAX_VALUE_ - MIN_VALUE_) * NUM_GRAMS_), 0), NUM_GRAMS_ - 1);
  }
}
