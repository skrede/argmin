#include "oracle/oracle.h"

int main()
{
    return argmin::oracle::claim_of(argmin::solver_status::converged) == argmin::oracle::claim::optimality ? 0 : 1;
}
