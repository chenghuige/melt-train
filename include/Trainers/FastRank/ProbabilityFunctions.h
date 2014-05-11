/**
 *  ==============================================================================
 *
 *          \file   ProbabilityFunctions.h
 *
 *        \author   chenghuige
 *
 *          \date   2014-05-10 10:09:39.384963
 *
 *  \Description:
 *  ==============================================================================
 */

#ifndef PROBABILITY_FUNCTIONS_H_
#define PROBABILITY_FUNCTIONS_H_
#include "common_util.h"
namespace gezi {

	class ProbabilityFunctions
	{
	private:
		static dvec _ProbA;
		static dvec _ProbB;
		static dvec _ProbC;
		static dvec _ProbD;
		static dvec _ProbE;
		static dvec _ProbF;

		ProbabilityFunctions()
		{
			_ProbA = { 3.3871328727963665, 133.14166789178438, 1971.5909503065513, 13731.693765509461, 45921.95393154987, 67265.7709270087, 33430.575583588128, 2509.0809287301227 };
			_ProbB = { 42.313330701600911, 687.18700749205789, 5394.1960214247511, 21213.794301586597, 39307.895800092709, 28729.085735721943, 5226.4952788528544 };
			_ProbC = { 1.4234371107496835, 4.6303378461565456, 5.769497221460691, 3.6478483247632045, 1.2704582524523684, 0.24178072517745061, 0.022723844989269184, 0.00077454501427834139 };
			_ProbD = { 2.053191626637759, 1.6763848301838038, 0.6897673349851, 0.14810397642748008, 0.015198666563616457, 0.00054759380849953455, 1.0507500716444169E-09 };
			_ProbE = { 6.6579046435011033, 5.4637849111641144, 1.7848265399172913, 0.29656057182850487, 0.026532189526576124, 0.0012426609473880784, 2.7115555687434876E-05, 2.0103343992922881E-07 };
			_ProbF = { 0.599832206555888, 0.13692988092273581, 0.014875361290850615, 0.00078686913114561329, 1.8463183175100548E-05, 1.4215117583164459E-07, 2.0442631033899397E-15 };
		}

	public:
		static double Erf(double x)
		{
			if (std::isinf(x))
			{
				if (x != std::numeric_limits<double>::infinity())
				{
					return -1.0;
				}
				return 1.0;
			}
			double t = 1.0 / (1.0 + (0.3275911 * std::abs(x)));
			double ev = 1.0 - ((((((((((1.061405429 * t) + -1.453152027) * t) + 1.421413741) * t) + -0.284496736) * t) + 0.254829592) * t) * std::exp(-(x * x)));
			if (x < 0.0)
			{
				return -ev;
			}
			return ev;
		}

		static double Erfc(double x)
		{
			if (std::isinf(x))
			{
				if (x != std::numeric_limits<double>::infinity())
				{
					return -1.0;
				}
				return 1.0;
			}
			double t = 1.0 / (1.0 + (0.3275911 * std::abs(x)));
			double ev = (((((((((1.061405429 * t) + -1.453152027) * t) + 1.421413741) * t) + -0.284496736) * t) + 0.254829592) * t) * std::exp(-(x * x));
			if (x < 0.0)
			{
				return (2.0 - ev);
			}
			return ev;
		}

		static double Erfinv(double x)
		{
			if (x == 1.0)
			{
				return std::numeric_limits<double>::infinity();
			}
			if (x == -1.0)
			{
				return -std::numeric_limits<double>::infinity();
			}
			dvec c(0x3e8);
			c[0] = 1.0;
			for (size_t k = 1; k < c.size(); k++)
			{
				for (int m = 0; m < k; m++)
				{
					c[k] += ((c[m] * c[(k - 1) - m]) / ((double)(m + 1))) / ((double)((m + m) + 1));
				}
			}
			double cc = std::sqrt(3.1415926535897931) / 2.0;
			double ccinc = 0.78539816339744828;
			double zz = x;
			double zzinc = x * x;
			double ans = 0.0;
			for (size_t k = 0; k < c.size(); k++)
			{
				ans += ((c[k] * cc) * zz) / ((double)((2 * k) + 1));
				cc *= ccinc;
				zz *= zzinc;
			}
			return ans;
		}

		static double Probit(double p)
		{
			double q = p - 0.5;
			double r = 0.0;
			if (std::abs(q) <= 0.425)
			{
				r = 0.180625 - (q * q);
				return ((q * ((((((((((((((_ProbA[7] * r) + _ProbA[6]) * r) + _ProbA[5]) * r) + _ProbA[4]) * r) + _ProbA[3]) * r) + _ProbA[2]) * r) + _ProbA[1]) * r) + _ProbA[0])) / ((((((((((((((_ProbB[6] * r) + _ProbB[5]) * r) + _ProbB[4]) * r) + _ProbB[3]) * r) + _ProbB[2]) * r) + _ProbB[1]) * r) + _ProbB[0]) * r) + 1.0));
			}
			if (q < 0.0)
			{
				r = p;
			}
			else
			{
				r = 1.0 - p;
			}
			if (r < 0.0)
			{
				THROW("Illegal input value");
			}
			r = std::sqrt(-std::log(r));
			double retval = 0.0;
			if (r < 5.0)
			{
				r -= 1.6;
				retval = ((((((((((((((_ProbC[7] * r) + _ProbC[6]) * r) + _ProbC[5]) * r) + _ProbC[4]) * r) + _ProbC[3]) * r) + _ProbC[2]) * r) + _ProbC[1]) * r) + _ProbC[0]) / ((((((((((((((_ProbD[6] * r) + _ProbD[5]) * r) + _ProbD[4]) * r) + _ProbD[3]) * r) + _ProbD[2]) * r) + _ProbD[1]) * r) + _ProbD[0]) * r) + 1.0);
			}
			else
			{
				r -= 5.0;
				retval = ((((((((((((((_ProbE[7] * r) + _ProbE[6]) * r) + _ProbE[5]) * r) + _ProbE[4]) * r) + _ProbE[3]) * r) + _ProbE[2]) * r) + _ProbE[1]) * r) + _ProbE[0]) / ((((((((((((((_ProbF[6] * r) + _ProbF[5]) * r) + _ProbF[4]) * r) + _ProbF[3]) * r) + _ProbF[2]) * r) + _ProbF[1]) * r) + _ProbF[0]) * r) + 1.0);
			}
			if (q < 0.0)
			{
				return -retval;
			}
			return retval;
		}
	};

}  //----end of namespace gezi

#endif  //----end of PROBABILITY_FUNCTIONS_H_
