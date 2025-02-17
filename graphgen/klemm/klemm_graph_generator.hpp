//
// Created by nathaniel on 4/24/18.
//

#ifndef GRAPHGEN_KLEMM_GRAPH_GENERATOR_HPP
#define GRAPHGEN_KLEMM_GRAPH_GENERATOR_HPP

#include <Python.h>
#include <deque>
#include <map>
#include <set>

PyObject* ConvertMemberDequeToTuple(const std::deque<std::deque<int> >& member_list);
PyObject* ConvertWeightMapToNumpyArray(const std::deque<std::map<int, double> >& Wout,
                                       const std::deque<std::set<int> >& Eout);
PyObject* ConvertEdgeDequeToNumpyArray(const std::deque<std::set<int> >& Eout);
PyMODINIT_FUNC PyInit_klemm_graph_generator(void);

#endif //GRAPHGEN_KLEMM_GRAPH_GENERATOR_HPP
