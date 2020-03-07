#include <torch/extension.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <algorithm>

torch::Tensor wrap(torch::Tensor & arr, int rec_stop, int left_index, int right_index, int rec_index){
    // termination?
    if (rec_index == rec_stop){
        return arr;
    }

    if (left_index >= right_index){
        return arr;
    }

    torch::Tensor arr_slice = at::slice(arr, -1, left_index, right_index);
    int num_elements = right_index - left_index;
    int top_high = (right_index - left_index) / 2;
    int num_r_elements = num_elements - top_high;

    int maxim = 0;

    // termination ?
    if (top_high > num_r_elements){
      maxim = top_high;
    }
    else{
      maxim = num_r_elements;
    }

    if (maxim > at::size(arr_slice, -1)){
      return arr;
    }

    int middle = left_index + top_high;
    for (int bs=0; bs<at::size(arr_slice, 0); bs++){
      auto c_arr = arr_slice[bs].data<long>();
      std::nth_element(c_arr, c_arr + top_high, c_arr + at::size(arr, -1));
    }

    torch::Tensor l_slice = at::slice(arr_slice, -1, left_index, left_index + top_high);
    torch::Tensor r_slice = at::slice(arr_slice, -1, top_high, right_index);

    // repeat for smaller part
    torch::Tensor ll = wrap(l_slice, rec_stop, left_index, middle, rec_index + 1);

    // repeat for larger part
    torch::Tensor rr = wrap(r_slice, rec_stop, middle, right_index, rec_index + 1);
    return arr;
}

// std::pair<torch::Tensor, torch::Tensor> long_sort(torch::Tensor arr, int num_buckets){
torch::Tensor long_sort(torch::Tensor arr, int num_buckets){
    int rec_stop = log2(num_buckets);

    // make a copy of the original array
    torch::Tensor s_arr = arr;
    // sort copy
    s_arr = wrap(s_arr, rec_stop, 0, at::size(arr, -1), 0);

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
