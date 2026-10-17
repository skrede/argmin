#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_WRAPPER_SLOTS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_WRAPPER_SLOTS_H

#include <nanobind/nanobind.h>

namespace argmin::python
{

template <typename Wrapper>
int wrapper_traverse(PyObject* self, visitproc visit, void* arg)
{
    Py_VISIT(Py_TYPE(self));
    if(!nanobind::inst_ready(self))
        return 0;
    for(PyObject* held : nanobind::inst_ptr<Wrapper>(self)->callable_handles())
        Py_VISIT(held);
    return 0;
}

template <typename Wrapper>
int wrapper_clear(PyObject* self)
{
    nanobind::inst_ptr<Wrapper>(self)->release_callables();
    return 0;
}

// Without these two the collector cannot break a cycle that runs through a
// held callable, and every such solver leaks for the interpreter's lifetime.
template <typename Wrapper>
inline PyType_Slot wrapper_slots[]{
    {Py_tp_traverse, reinterpret_cast<void*>(&wrapper_traverse<Wrapper>)},
    {Py_tp_clear, reinterpret_cast<void*>(&wrapper_clear<Wrapper>)},
    {0, nullptr}};

}

#endif
