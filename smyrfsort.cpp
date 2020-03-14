#include <torch/extension.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <algorithm>
#include <stack>
#include <utility>

void print_array(torch::Tensor arr){
  int N = size(arr, -1);
  long * c_arr = arr.data<long>();
  for (int i=0; i<N; i++){
    std::cout << " " << c_arr[i] << " ";
  }
  std::cout << std::endl;
}


torch::Tensor wrap(torch::Tensor & arr, int rec_stop){
    std::stack<std::pair<int, int>> todos;
    int N = size(arr, -1);

    todos.push(std::make_pair(0, N - 1));
    long * c_arr = arr.data<long>();

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

        // leave the right part for later
        todos.push(std::make_pair(left + n2_biggest, right));

        // continue with the left part
        right = n2_biggest - 1;
      }

    }
    return arr;
}

// std::pair<torch::Tensor, torch::Tensor> long_sort(torch::Tensor arr, int num_buckets){
torch::Tensor long_sort(torch::Tensor arr, int num_buckets){
    int rec_stop = log2(num_buckets);

    // make a copy of the original array
    torch::Tensor s_arr = arr;
    // sort copy
    s_arr = wrap(s_arr, rec_stop);

    return s_arr;
    //
    // std::map<long, std::vector<long>> indices_map;
    // std::map<long, std::vector<long>>::iterator it = indices_map.begin();
    // std::vector<long>::iterator vec_it;
    //
    // for (int bs=0; bs<at::size(s_arr, 0); bs++){
    //   for (int i=0; i<at::size(s_arr, -1); i++){
    //     long elem = *(s_arr[bs][i].data<long>());
    //     it = indices_map.find(elem);
    //     if (it != indices_map.end()){
    //       // found -> append
    //       (*it).second.push_back(i);
    //     }
    //     else{
    //       // not found -> initialize
    //       std::vector<long> vec;
    //       vec.push_back(i);
    //       indices_map.insert(std::make_pair(elem, vec));
    //     }
    //   }
    // }
    //
    // torch::Tensor sorted_indices = torch::empty_like(arr, torch::dtype(torch::kLong));
    //
    // for (int bs=0; bs<at::size(arr, 0); bs++){
    //   for (int i=0; i<at::size(arr, -1); i++){
    //     long elem = *(arr[bs][i].data<long>());
    //     it = indices_map.find(elem);
    //     for (vec_it=(*it).second.begin(); vec_it!=(*it).second.end(); vec_it++){
    //       long * data_ptr = sorted_indices[bs][i].data<long>();
    //       (*data_ptr) = *vec_it;
    //     }
    //   }
    // }
    //
    // // std:: cout << arr[0][0] << std:: endl;
    // // std:: cout << s_arr[0][0] << std:: endl;
    // // std::cout << sorted_indices[0][0] << std::endl;
    // return std::make_pair(s_arr, sorted_indices);
  }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("long", &long_sort, "long");
}
