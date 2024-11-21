#ifndef PRINT_ARR_H
#define PRINT_ARR_H

#include<iostream>
#include<vector>
using namespace std;
/*template <typename T> //allows this function to workwith any datatype  
void print_array(const T* arr,int size){ //param: T ptr to array of type
    for(int i = 0;i < size;i++){
        os<<arr[i]<<" ";
    }
    os<<"\n";
}

template <typename T>
void print_vec(const vector<T>& vec){
    for(auto &i:vec){
        os<<i<<" ";
    }
    os<<"\n";
}
template<typename T>
void print_matrix(const vector<T> &vec){
    for (int i = 0;i < vec.size();i++){
        for(auto &j:vec[i]){
            os << j << " ";
        }
        os<<"\n";
    }
    os << "\n";
}
template <typename T>
void print_array(const T* ar,int row,int col){
    for(int i = 0;i < row;i++){
        for(int j = 0;j < col;j++){
            os<<ar[i][j]<<" ";
            //os<<*((ar + i * col) + j)<<" ";
        }
        os<<"\n";
    }
}

template <typename T>
void reverse_print_array(const T *arr, int size)
{
    for (int i = size-1; i >= 0 ; i--){
        os << arr[i] << " ";
    }
    os << "\n";
}*/

// pass a stream (std::ostream) as an argument This can redirect the output to a file or a faster stream
template <typename T>
void print(const T& container,ostream& os = cout){
    for(const auto &i:container){
         os<<i<<" ";
    }
    os<<"\n";
}

//for C -type arrays
template <typename T, size_t N>
void print(const T (&arr)[N], ostream &os = cout)
{
    for(const auto &i:arr){
        os<<i<<" ";
    }
    os<<"\n";
}

//2-D arrays
template <typename T, size_t r, size_t c>
void print(const T (&arr)[r][c], ostream &os = cout)
{
    for(size_t i = 0;i < r;i++){
        for(size_t j = 0;j < c;j++){
            os<<arr[i][j]<<" ";
        }
        os<<"\n";
    }
    os<<"\n";
}
template <typename T>
void print_matrix(const vector<T> &vec, ostream &os = cout)
{
    for (int i = 0; i < vec.size(); i++)
    {
        for (auto &j : vec[i])
        {
            os << j << " ";
        }
        os << "\n";
    }
    os << "\n";
}
#endif 