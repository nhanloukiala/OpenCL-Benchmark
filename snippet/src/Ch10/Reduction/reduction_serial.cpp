#include <cstdlib>
#include <iostream>

template<typename T>
T reduce(T (*f)(T, T), 
         size_t n,
         T a[],
         T identity)
{
  T accum = identity;
  for(size_t i = 0; i < n ; ++i) 
        accum = f(accum, a[i]);
  return accum;
}

int add(int a, int b) {return a + b;}
int identity = 0;
int main(int argc, char** argv) {
       int a[100];
       for(int i = 0; i < 100; ++i) a[i] = i+1;
       int sum = reduce<int>(add, 100, a, identity); 
       std::cout << "Total sum = " << sum << std::endl;
}

