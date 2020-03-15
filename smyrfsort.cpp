#include <torch/extension.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <algorithm>
#include <stack>
#include <utility>
#include <google/dense_hash_map>

void print_array(torch::Tensor arr){
  int N = size(arr, -1);
  long * c_arr = arr.data<long>();
  for (int i=0; i<N; i++){
    std::cout << " " << c_arr[i] << " ";
  }
  std::cout << std::endl;
}


torch::Tensor wrap(torch::Tensor arr, int rec_stop){
    int bs = size(arr, 0);
    for (int i=0; i<bs; i++){
      std::stack<std::pair<int, int>> todos;
      int N = size(arr, -1);

      todos.push(std::make_pair(0, N - 1));
      long * c_arr = arr[i].data<long>();

      int rec_index = 0;
      while (!todos.empty()){
        std::pair<int, int> top = todos.top();
        todos.pop();
        int left = std::get<0>(top);
        int right = std::get<1>(top);

        while (left < right){
          int num_elements = right - left + 1;

          // if unequal give it to the right side.
          int n2_biggest = num_elements / 2;

          // split into two halves
          std::nth_element(c_arr + left, c_arr + left + n2_biggest, c_arr + right + 1);

          if (rec_index < rec_stop){
            // leave the right part for later
            todos.push(std::make_pair(left + n2_biggest, right));
            // continue with the left part
            right = n2_biggest - 1;
          }
          else{
            break;
          }

          if (left == 0) rec_index += 1;
        }

      }
    }
    return arr;
}

std::pair<torch::Tensor, torch::Tensor> long_sort(torch::Tensor arr, int num_buckets){

    int rec_stop = log2(num_buckets);
    int batch_size = size(arr, 0);
    int dim = size(arr, -1);

    // {'value1': [index1, index2, ...], }
    torch::Tensor sorted_indices = torch::empty_like(arr, torch::dtype(torch::kLong));

    google::dense_hash_map<long, std::vector<long>> indices_map;
    indices_map.set_empty_key(NULL);
    std::vector<long>::iterator vec_it;


    for (int bs=0; bs<batch_size; bs++){
      long * c_arr = arr[bs].data<long>();
      for (int i=0; i<dim; i++){
        long elem = c_arr[i];
        indices_map[elem].push_back(i);
      }
    }

    // sort copy
    wrap(arr, rec_stop);



    // map each value to the indices it appears
    for (int bs=0; bs<batch_size; bs++){
      long * elems = arr[bs].data<long>();
      long * s_indices = sorted_indices[bs].data<long>();

      for (int i=0; i<dim; i++){
        long elem = elems[i];
        std::vector<long> found = indices_map[elem];

        if (found.empty()){
          // how this happens?
          std::cout << "Warning. Something is wrong here." << std::endl;
          found.push_back(i);
        }

        // append index in which this value appears
        s_indices[i] = found.back();
        // remove this index
        found.pop_back();
      }
    }

    return std::make_pair(arr, sorted_indices);

  }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sort", &long_sort, "sort");
}
