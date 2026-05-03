#include "argmin/argmin.h"

int main()
{
    argmin::vector<double> v(3);
    v << 1.0, 2.0, 3.0;
    return v.size() == 3 ? 0 : 1;
}
