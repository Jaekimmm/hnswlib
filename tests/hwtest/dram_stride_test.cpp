#include <iostream>
#include <chrono>
#include <vector>

#define ARRAY_SIZE 1024 * 1024 * 64  // 64 MB
#define STRIDE 64                    // Cache Line Size (64 Bytes)

int main() {
    std::vector<int> arr(ARRAY_SIZE, 1); // Allocate large array
    int sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Access memory with stride
    for (size_t i = 0; i < arr.size(); i += STRIDE / sizeof(int)) {
        sum += arr[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Stride: " << STRIDE << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "Sum: " << sum << std::endl;

    return 0;
}
