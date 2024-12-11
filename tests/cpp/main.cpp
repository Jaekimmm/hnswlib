#include <vector>

void sift_test1B(int subset_size_milllions);
std::vector<int> data_size_millions;

int main() {
    
    //* datasize sweep test
//    std::vector<int> data_size_millions = {1, 2, 5, 10, 20, 50, 100, 200, 500};

//    for (int n : data_size_millions) {
        sift_test1B(1);
//    }

    return 0;
}
