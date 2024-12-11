#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

class StopW {   //* stopwatch
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {   //* capture currnet time
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() { //* calculate duration
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {  //* stopwatch reset to current time
        time_begin = std::chrono::steady_clock::now();
    }
};



/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


static void
get_gt(                     //* get answer index from ground-truth file
    unsigned int *massQA,   //* answer index list
    unsigned char *massQ,   //* pointer of query data start addr
    unsigned char *mass,    //* pointer of vector data start addr
    size_t vecsize,         //* # of vector data
    size_t qsize,           //* # of query data
    L2SpaceI &l2space,      //* object of calculating L2-distance
    size_t vecdim,          //* vector dimension
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,  //* object of final answer list
    size_t k) {             //* top-k
    (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
    //* create empty priority_quere with the size of query vector
    //* swap this with answers
    //*  --> then answers (final answer list) will be initialized to empty vector
    DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();     //* get dist function from l2space opject (dummy code)
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);   //* Get top-k index of each query from gt file
        }
    }
}

static float
test_approx(          //* run ANN search and compare with gt
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;
    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {           //* For each query,
        std::priority_queue<std::pair<int, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);  //* do HNSW search
        std::priority_queue<std::pair<int, labeltype >> gt(answers[i]);   //* get answers of the query from gt
        unordered_set<labeltype> g;             //* create unordered set 'g' to store the query's gt answer
        total += gt.size();                     //* accumulate # of answers 

        while (gt.size()) {                     //* For all gt answers
            g.insert(gt.top().second);          //* put the first answer's index to 'g' set
            gt.pop();                           //* and pop it
        }

        while (result.size()) {                 //* For all ANN results,
            if (g.find(result.top().second) != g.end()) { //* Find the index along gt index set
                correct++;                              //* If there's matched index, count up correct
            } else {
            }
            result.pop();                               //* and pop it
        }
    }
    return 1.0f * correct / total;              //* Return R@K
}

static void
test_vs_recall(                            //* Sweep test (ef --> qps & r@k)
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    vector<size_t> efs;  // = { 10,10,10,10,10 };
    //* generate ef testcase
    for (int i = k; i < 30; i++) {
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10) {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 50) {
        efs.push_back(i);
    }
//* JW code start (save test result as .csv file)
    int size_m = vecsize / 1000000;
    std::string outfile = "test_" + std::to_string(size_m) + "M_top" + std::to_string(k) + ".csv";
    std::ofstream fout (outfile);
    if (fout.is_open()) {
         fout << "ef,r@k,runtime(us),qps\n";
//* JW code end
    for (size_t ef : efs) {                     //* For each ef test case,
        appr_alg.setEf(ef);                     //* set ANN ef config
        StopW stopw = StopW();                  //* Stopwatch start to get search_time

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);    //* run test and get r@k
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;  //* Calculate qps

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
        float qps = 1000000 / time_us_per_query;
        fout << ef << "," << recall << "," << time_us_per_query << "," << qps << "\n";
    }
//* jw code start (save test result as .csv file -> file close)
    fout.close();
    }
//* jw code end
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


//* void sift_test1B() {
void sift_test1B(int subset_size_milllions) {
    //* int subset_size_milllions = 200;
    //*int subset_size_milllions = 10;  //* as kim et al. single sub-graph size for 4GB DRAM is 5M
    int efConstruction = 40;
    int M = 16;

    size_t vecsize = subset_size_milllions * 1000000;

    size_t qsize = 10000;
    size_t vecdim = 128;
    char path_index[1024];
    char path_gt[1024];
    const char *path_q = "../bigann/bigann_query.bvecs";
    const char *path_data = "../bigann/bigann_base.bvecs";
    snprintf(path_index, sizeof(path_index), "sift1b_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);

    snprintf(path_gt, sizeof(path_gt), "../bigann/gnd/idx_%dM.ivecs", subset_size_milllions);

    unsigned char *massb = new unsigned char[vecdim];

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            cout << "err";
            return;
        }
    }
    inputGT.close();

    cout << "Loading queries:\n";
    unsigned char *massQ = new unsigned char[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        inputQ.read((char *) massb, in);
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }
    }
    inputQ.close();


    unsigned char *mass = new unsigned char[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    L2SpaceI l2space(vecdim);

    HierarchicalNSW<int> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<int>(&l2space, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<int>(&l2space, vecsize, M, efConstruction);

        input.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        input.read((char *) massb, in);

        for (int j = 0; j < vecdim; j++) {
            mass[j] = massb[j] * (1.0f);
        }

        appr_alg->addPoint((void *) (massb), (size_t) 0);
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
            unsigned char mass[128];
            int j2 = 0;
#pragma omp critical
            {
                input.read((char *) &in, 4);
                if (in != 128) {
                    cout << "file error";
                    exit(1);
                }
                input.read((char *) massb, in);
                for (int j = 0; j < vecdim; j++) {
                    mass[j] = massb[j];
                }
                j1++;
                j2 = j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) (mass), (size_t) j2);
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
    }


    vector<std::priority_queue<std::pair<int, labeltype >>> answers;
//*    size_t k = 1;
    size_t k = 10;
    cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, l2space, vecdim, answers, k);
    cout << "Loaded gt\n";
    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return;
}
