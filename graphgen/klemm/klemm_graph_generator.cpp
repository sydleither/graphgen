/*
 * Provides access in Python to LFR graphs generated from c++ code
 */

#include <Python.h>
#include <cstddef>
#include <deque>
#include <set>
#include <map>
#include <iostream>
#include <vector>
#include "numpy/arrayobject.h"
#include "benchm.hpp"
#include "klemm_graph_generator.hpp"

static PyObject *GenerateKlemmGraph(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"num_nodes", "clique_size", "clique_linkage",
                                 "muw", "beta", "seed"};

  int num_nodes;
  int clique_size;
  double clique_linkage;
  double muw;
  double beta;
  int seed;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iidddi", keyword_list,
                                   &num_nodes, &clique_size, &clique_linkage,
                                   &muw, &beta, &seed)) {
    std::cerr << "Error parsing GenerateKlemmGraph arguments" << std::endl;
    return NULL;
  }

  std::deque<std::set<int> > Eout;
  std::deque<std::deque<int> > member_list;
  std::deque<std::map<int, double> > Wout;
  build_network_klemm(num_nodes, clique_size, clique_linkage, muw, beta, seed, Eout, member_list, Wout);

  PyObject* edge_array = ConvertEdgeDequeToNumpyArray(Eout);
  PyObject* weight_array = ConvertWeightMapToNumpyArray(Wout, Eout);
  PyObject* member_tuple = ConvertMemberDequeToTuple(member_list);
  PyObject* return_tuple = PyTuple_New(3);
  PyTuple_SetItem(return_tuple, 0, edge_array);
  PyTuple_SetItem(return_tuple, 1, member_tuple);
  PyTuple_SetItem(return_tuple, 2, weight_array);

  return return_tuple;
}

PyObject* ConvertEdgeDequeToNumpyArray(const std::deque<std::set<int> >& Eout) {
  // Determine the number of edges for the array
  npy_intp num_edges(0);
  for (auto head_set = Eout.begin(); head_set != Eout.end(); ++head_set) {
    for (auto head = head_set->begin(); head != head_set->end(); ++head) {
      ++num_edges;
    }
  }

  // Generate dimensional information for Ex2 edge array
  std::vector<npy_intp> size_data(2);
  size_data[0] = num_edges;
  size_data[1] = 2;

  // Create new numpy array
  PyObject* py_array = PyArray_SimpleNew(2, size_data.data(), NPY_UINT64);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_uint64* data = reinterpret_cast<npy_uint64*>(np_array->data);

  // Fill numpy array with edges
  std::size_t edge_index(0);
  for (std::size_t tail = 0; tail < Eout.size(); ++tail) {
    for (auto head = Eout[tail].begin(); head != Eout[tail].end(); ++head) {
      data[edge_index] = tail;
      data[edge_index+1] = *head;
      edge_index += 2;
    }
  }

  return py_array;
}

PyObject* ConvertWeightMapToNumpyArray(const std::deque<std::map<int, double> >& Wout,
                                       const std::deque<std::set<int> >& Eout) {
  // Determine the number of edges for the array
  npy_intp num_edges(0);
  for (auto head_set = Eout.begin(); head_set != Eout.end(); ++head_set) {
    for (auto head = head_set->begin(); head != head_set->end(); ++head) {
      ++num_edges;
    }
  }

  // Generate dimensional information for weight array
  std::vector<npy_intp> size_data(1);
  size_data[0] = num_edges;

  // Create new numpy array
  PyObject* py_array = PyArray_SimpleNew(1, size_data.data(), NPY_FLOAT32);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_float32* data = reinterpret_cast<npy_float32*>(np_array->data);

  // Fill numpy array with edges
  std::size_t edge_index(0);
  for (std::size_t tail = 0; tail < Eout.size(); ++tail) {
    for (auto head = Eout[tail].begin(); head != Eout[tail].end(); ++head) {
      data[edge_index] = static_cast<npy_float32>(Wout[tail].at(*head));
      edge_index += 1;
    }
  }

  return py_array;
}

PyObject* ConvertMemberDequeToTuple(const std::deque<std::deque<int> >& member_list) {

  PyObject* member_tuple = PyTuple_New(static_cast<Py_ssize_t>(member_list.size()));

  // For each node loop through and make a tuple of its members
  for (Py_ssize_t member = 0; member < PyTuple_Size(member_tuple); ++member) {
    PyObject* communities_tuple = PyTuple_New(static_cast<Py_ssize_t>(
                                              member_list[member].size()));

    // Add community memberships to nodes tuple
    for (Py_ssize_t iii = 0; iii < PyTuple_Size(communities_tuple); ++iii) {
      PyTuple_SetItem(communities_tuple, iii, PyLong_FromLong(
      static_cast<long>(member_list[member][iii])));
    }

    PyTuple_SetItem(member_tuple, member, communities_tuple);
  }

  return member_tuple;
}

static PyMethodDef KlemmGeneratorMethods[] = {
{ "GenerateKlemmGraph", (PyCFunction) GenerateKlemmGraph,
              METH_VARARGS | METH_KEYWORDS,
"Creates Klemm-Eguiliuz graphs with LFR weights. Returns Ex2 numpy edge "
"array and tuple of community assignments"},
{ NULL, NULL, 0, NULL}
};

static struct PyModuleDef KlemmGeneratorModule = {
PyModuleDef_HEAD_INIT,
"klemm_graph_generator",
"Creates Klemm-Eguiliuz graphs with LFR weights",
-1,
KlemmGeneratorMethods
};

PyMODINIT_FUNC PyInit_klemm_graph_generator(void) {
  import_array();
  return PyModule_Create(&KlemmGeneratorModule);
}