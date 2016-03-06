//
// Created by yanhang on 2/13/16.
//

#include "utility.h"

namespace math_util{
    double normalizedCrossCorrelation(const std::vector<double> &a1, const std::vector<double> &a2) {
            CHECK_EQ(a1.size(), a2.size());
            CHECK(!a1.empty());
            const double n = (double)a1.size();
            double m1 = std::accumulate(a1.begin(), a1.end(), 0.0) / n;
            double m2 = std::accumulate(a2.begin(), a2.end(), 0.0) / n;
            double var1 = math_util::variance(a1, m1);
            double var2 = math_util::variance(a2, m2);
	    if(var1 == 0 || var2 == 0 )
		return 0;
		
            double ncc = 0;
            for (size_t i = 0; i < a1.size(); ++i)
                    ncc += (a1[i] - m1) * (a2[i] - m2);
            ncc /= var1 * var2 * n;
            CHECK_LE(ncc, 1.001);
            CHECK_GE(ncc, -1.001);
            return ncc;
    }

    double SSDScore(const std::vector<double>& a1, const std::vector<double>& a2){
        CHECK_EQ(a1.size(), a2.size());
        if(a1.empty())
            return 0.0;
        double ssd = 0.0;
        for(size_t i=0; i<a1.size(); ++i)
            ssd += (a1[i] - a2[i]) * (a1[i] - a2[i]);
        return ssd /(double)a1.size();
    }
}
