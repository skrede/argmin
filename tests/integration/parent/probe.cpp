#include "transitive_parent/probe.h"

#include "argmin/argmin.h"

namespace transitive_parent
{

double probe()
{
    argmin::vector<double> v(3);
    v << 1.0, 2.0, 3.0;
    return v.sum();
}

}
