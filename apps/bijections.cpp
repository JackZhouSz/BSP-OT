#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "../src/BSPOTWrapper.h"
#include "../src/cloudutils.h"
#include "../src/sampling.h"
#include "../src/PointCloudIO.h"
#include "../src/PointCloudIO.h"
#include "../src/plot.h"
#include "../src/sliced.h"
#include <queue>
#include <omp.h>
#include <thread>
#include <chrono>
#include <condition_variable>
#include "../common/discrete_OT_solver.h"
#include "../common/CLI11.hpp"


int N = 1000;
int nb_plans = 16;

constexpr int static_dim = 2;
int dim = static_dim;

using namespace BSPOT;

using Pts = Points<static_dim>;

scalar eval(const Pts& A,const Pts& B,const ints& T) {
    return std::sqrt((A - B(Eigen::all,T)).squaredNorm());
}

Pts A,B;

inline scalar cost(int i,int j) {
    return (A.col(i) - B.col(j)).squaredNorm();
}

BijectiveMatching T,Tex;

Vec evalMappings(const BijectiveMatching& T) {
    // compute square distance between A and B(Eigen::all,T)
    Vec costs(T.size());
    for (auto i : range(T.size()))
        costs[i] = (A.col(i) - B.col(T[i])).squaredNorm();
    // Pts diff = A - B(Eigen::all,T);
    // Vec costs = diff.colwise().squaredNorm().transpose();
    return costs;
}

ints rankPlans(const std::vector<BijectiveMatching>& plans) {
    auto start = Time::now();
    std::vector<std::pair<scalar,int>> scores(plans.size());
    for (auto i : range(plans.size())) {
        scores[i].first = evalMappings(T).sum();
        scores[i].second = i;
    }
    std::sort(scores.begin(),scores.end(),[](const auto& a,const auto& b) {
        return a.first < b.first;
    });
    // spdlog::info("sort timing {}",TimeFrom(start));
    ints rslt(scores.size());
    for (auto i : range(scores.size()))
        rslt[i] = scores[i].second;
    return rslt;
}


BijectiveMatching MergePlansNoPar(const std::vector<BijectiveMatching> &plans, BijectiveMatching T,bool cycle) {
    int s = 0;
    auto I = true ? rankPlans(plans) : rangeVec(plans.size());
    if (T.size() == 0) {
        T = plans[I[0]];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T);

    ints rslt = T;
    ints rsltI = T.inverseMatching();

    ints sig(N);

    StopWatch profiler;
    for (auto k : range(s,plans.size())) {
        ints Tp = plans[I[k]];
        ints Tpi = plans[I[k]].inverseMatching();
        auto Cp = evalMappings(Tp);

        for (auto i : range(N))
            sig[i] = Tpi[rslt[i]];

        // profiler.start();

        std::vector<ints> CCs;

        if (cycle) {
            ints visited(N,-1);
            int c = 0;
            for (auto i : range(N)) {
                if (visited[i] != -1)
                    continue;
                int j = i;
                int i0 = i;
                if (sig[j] == i)
                    continue;

                ints CC;
                scalar costT = 0;
                scalar costTP = 0;

                while (visited[j] == -1) {
                    CC.push_back(j);
                    costT  += C[j];
                    costTP += Cp[j];
                    visited[j] = c;
                    j = sig[j];
                }

                if (costTP < costT) {
                    j = i0;
                    do {
                        std::swap(Tp[j],rslt[j]);
                        std::swap(C[j],Cp[j]);
                        j = sig[j];
                    } while (j != i0);
                    j = i0;
                    do {
                        rsltI[rslt[j]] = j;
                        j = sig[j];
                    } while (j != i0);
                }

                c++;
                CCs.push_back(CC);
            }
        }
        for (int i = 0; i < CCs.size(); ++i) {
            {
                for (auto a : CCs[i]){
                    // swapIfUpgradeK(rslt,rsltI,Tp,a,3,cost);
                    int b = rslt[a];
                    int bp = Tp[a];
                    int ap  = rsltI[bp];
                    if (a == ap || b == bp)
                        continue;
                    scalar old_cost = C[a] + C[ap];
                    scalar cabp = Cp[a];
                    if (cabp > old_cost)
                        continue;
                    scalar capb = cost(ap,b);
                    if (cabp + capb < old_cost) {
                        rslt[a] = bp;
                        rslt[ap] = b;
                        rsltI[bp] = a;
                        rsltI[b] = ap;
                        C[a] = cabp;
                        C[ap] = capb;
                    }
                }
            }
        }
    }
    return rslt;
}

scalar noise = 0e-3;

void compute() {
    auto plans = std::vector<BijectiveMatching>(nb_plans);
    auto start = Time::now();
    BijectiveBSPMatching<static_dim> BSP(A,B);
#pragma omp parallel for
    for (auto& plan : plans) {
        plan = BSP.computeGaussianMatching();
//        plan = BSP.computeOrthogonalMatching(sampleUnitGaussianMat(dim,dim).fullPivHouseholderQr().matrixQ(),false);
//        plan = BSP.computeOrthogonalMatching(Q,false);
//        plan = BSP.computeMatching();
    }
    spdlog::info("compute time {}",TimeFrom(start));
    T = MergePlans(plans,cost,BijectiveMatching(),(N < 5e5));
//    Vec costs(nb_plans);
//    for (auto i : range(nb_plans)) {
//        T = Merge(T,plans[i],cost);
//        costs[i] = eval(A,B,T);
//    }
//    std::ofstream outfile("/tmp/costs.data");
//    outfile << costs;
    spdlog::info("merge time {} cost {}",TimeFrom(start),eval(A,B,T));
    start = Time::now();
    // T = MergePlans(plans,cost,BijectiveMatching(),(N < 5e5));
    T = MergePlansNoPar(plans,BijectiveMatching(),(N < 5e5));
    scalar w = eval(A,B,T);
    spdlog::info("merge time {} cost {}",TimeFrom(start),w);
}

Pts lerp(float t) {
    return Pts(A*(1-t) + t*B(Eigen::all,T));
}


float t = 0;
polyscope::PointCloud* pclerp = nullptr;
void myCallBack() {
    if (ImGui::SliderFloat("lerp",&t,0,1)){
        auto L = lerp(t);
        if (!pclerp)
            pclerp = display<static_dim>("lerp",L);
        else{
            if (L.rows() == 2)
                pclerp->updatePointPositions2D(L.transpose());
            else if (L.rows() == 3)
                pclerp->updatePointPositions(L.transpose());
        }
    }
}

int main(int argc,char** argv) {
    srand(time(NULL));
    if (std::is_same<scalar,double>::value) {
        spdlog::error("for the bijective matching, compiling using float may lead to a significant speed-up without affecting its quality \n you can change the type in 'common/types.h' ");
    }

    CLI::App app("BSPOT bijection") ;

    app.add_option("--sizes", N, "Number of samples in each cloud (default 1000)");
    app.add_option("--iter", nb_plans, "Number of iterations of refinement (default 100)");

    std::string mu_src;
    std::string nu_src;
    app.add_option("--mu_file", mu_src, "source cloud file (if empty then generated)");
    app.add_option("--nu_file", nu_src, "target cloud file (if empty then generated)");

    std::string plan_src;
    app.add_option("--plan_file", plan_src, "initial transport plan");

    std::string out_src;
    app.add_option("--output_file", out_src, "output transport plan");

    app.add_option("--noise", noise, "noise perturbation");

    if (static_dim == -1)
        app.add_option("--dim",dim,"dimension of the clouds, required if compiled with static_dim == -1")->required(true);

    bool viz = false;
    app.add_flag("--viz", viz, "use polyscope");
    bool force_size = false;
    app.add_flag("--force_size", force_size, "reduces the size of the measures to sizes if true");

    CLI11_PARSE(app, argc, argv);

    if (!mu_src.empty()){
        A = ReadPointCloud<static_dim>(mu_src);
        spdlog::info("mu | dim : {} size : {}",A.rows(),A.cols());
    } else {
        A = sampleUnitBall<static_dim>(N,dim);
    }

    if (!nu_src.empty()){
        B = ReadPointCloud<static_dim>(nu_src);
    } else {
        B = sampleUnitBall<static_dim>(A.cols(),dim);
    }

    if (!plan_src.empty()){
        T = load_plan(plan_src);
        if (T.size() != A.cols() || !T.checkBijectivity()){
            spdlog::error("invalid plan loaded {}",T.size());
            T = BijectiveMatching();
        }
    }

    if (force_size) {
        A = ForceToSize(A,N);
        B = ForceToSize(B,N);
    } else
        N = A.cols();

    Normalize<static_dim>(A);
    Normalize<static_dim>(B);

    compute();

    if (!out_src.empty())
        out_plan(out_src,T);

    if (viz) {
        polyscope::init();

        display<static_dim>("A",A);
        display<static_dim>("B",B);
        plotMatching("bspot",A,B,T)->setEnabled(false);

        polyscope::state::userCallback = myCallBack;
        polyscope::show();
    }
    return 0;
}
