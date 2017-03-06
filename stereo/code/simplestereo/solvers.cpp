//
// Created by yanhang on 3/10/16.
//

#include "../external/TRW_S/MRFEnergy.h"
#include "../stereo_base/depth.h"
#include "optimization.h"

#include "../external/QPBO1.4/QPBO.h"

using namespace std;
using namespace cv;
using namespace ParallelFusion;

namespace simple_stereo {

void SimpleStereoGenerator::getProposals(
    CompactLabelSpace &proposals, const CompactLabelSpace &current_solution,
    const int N) {
  for (auto i = 0; i < N; ++i) {
    proposals.getSingleLabel().push_back(labelTable[nextLabel]);
    nextLabel = (nextLabel + 1) % (int)labelTable.size();
  }
}

void SimpleStereoSolver::initSolver(const CompactLabelSpace &initial) {
  dataCost =
      shared_ptr<DataCost>(new DataCost(const_cast<int *>(model->MRF_data)));
  smoothnessCost = shared_ptr<SmoothnessCost>(new SmoothnessCost(
      1, 4, model->weight_smooth, const_cast<int *>(model->hCue),
      const_cast<int *>(model->vCue)));
  energy_function = shared_ptr<EnergyFunction>(
      new EnergyFunction(dataCost.get(), smoothnessCost.get()));
  mrf = shared_ptr<Expansion>(new Expansion(
      model->width, model->height, model->nLabel, energy_function.get()));
  mrf->initialize();
}

void SimpleStereoSolver::solve(
    const CompactLabelSpace &proposals,
    const SolutionType<CompactLabelSpace> &current_solution,
    SolutionType<CompactLabelSpace> &solution) {
  CHECK(!proposals.empty());
  int kFullProposal;
  if (proposals.getLabelSpace().empty())
    kFullProposal = 0;
  else
    kFullProposal = (int)proposals.getLabelOfNode(0).size();

  const vector<int> &singleLabel = proposals.getSingleLabel();

  solution = current_solution;
  for (auto i = 0; i < kPix; ++i)
    mrf->setLabel(i, current_solution.second(i, 0));
  for (auto i = 0; i < singleLabel.size(); ++i) {
    // printf("Fusing proposal with graph cut %d\n", singleLabel[i] );
    mrf->alpha_expansion(singleLabel[i]);
    // cout << "done" << endl << flush;
  }
  for (auto i = 0; i < kPix; ++i)
    solution.second(i, 0) = mrf->getLabel(i);

  for (auto i = 0; i < kFullProposal; ++i) {
    // run QPBO
    // printf("Running QPBO...\n");
    fuseTwoSolution(solution.second, proposals, i, model);
    // printf("Done. Unlabeled:%.3f, label changed:%.3f\n", unlabeled /
    // (float)kPix, changed / (float)kPix);
  }
  solution.second.getSingleLabel().clear();
  solution.first = evaluateEnergy(solution.second);
}

double
SimpleStereoSolver::evaluateEnergy(const CompactLabelSpace &solution) const {
  CHECK_EQ(solution.getNumNode(), kPix);
  double e = 0;
  const int w = model->width;
  const int h = model->height;
  const double r = model->MRFRatio;
  for (auto i = 0; i < kPix; ++i) {
    e += model->MRF_data[i * model->nLabel + solution(i, 0)] / model->MRFRatio;
  }
  for (auto x = 0; x < w - 1; ++x) {
    for (auto y = 0; y < h - 1; ++y) {
      int sc = model->computeSmoothCost(y * w + x, solution(y * w + x, 0),
                                        solution(y * w + x + 1, 0), true) +
               model->computeSmoothCost(y * w + x, solution(y * w + x, 0),
                                        solution((y + 1) * w + x, 0), false);
      e += (double)sc / r;
    }
  }
  return e;
}

void SimpleStereoMonitor::initSolver(const CompactLabelSpace &initial) {
  SimpleStereoSolver::initSolver(initial);
  start_time = (float)getTickCount();
}
void SimpleStereoMonitor::solve(
    const CompactLabelSpace &proposals,
    const ParallelFusion::SolutionType<CompactLabelSpace> &current_solution,
    ParallelFusion::SolutionType<CompactLabelSpace> &solution) {
  CHECK_EQ(proposals.getLabelSpace().size(), kPix);
  const int nProposals = (int)proposals.getLabelSpace()[0].size();
  if (depths_threads.empty())
    depths_threads.resize(nProposals);
  else
    CHECK_EQ(depths_threads.size(), nProposals);
  CompactLabelSpace bestS;
  double minEnergy = numeric_limits<double>::max();
  for (auto i = 0; i < nProposals; ++i) {
    stereo_base::Depth deptht;
    deptht.initialize(model->width, model->height, -1.0);
    for (auto j = 0; j < kPix; ++j)
      deptht.setDepthAtInd(j, proposals(j, i));
    depths_threads[i].push_back(deptht);

    CompactLabelSpace curs;

    curs.init(kPix, vector<int>(1, 0));
    for (auto j = 0; j < kPix; ++j)
      curs(j, 0) = proposals(j, i);
    double curE = evaluateEnergy(curs);
    if (curE < minEnergy) {
      minEnergy = curE;
      bestS = curs;
    }
  }
  float curt = ((float)getTickCount() - start_time) / (float)getTickFrequency();
  observations.push_back(Observation(curt, minEnergy));
  stereo_base::Depth depth;
  depth.initialize(model->width, model->height, -1.0);
  for (auto i = 0; i < kPix; ++i)
    depth.setDepthAtInd(i, bestS(i, 0));
  depths.push_back(depth);
}

void SimpleStereoMonitor::dumpData(const std::string &path) const {
  char buffer[1024] = {};
  printf("Monitor dumping out data...\n");
  CHECK_EQ(depths.size(), observations.size());
  sprintf(buffer, "%s/monitorProfile.txt", path.c_str());
  ofstream fout(buffer);
  CHECK(fout.is_open());
  for (auto i = 0; i < depths.size(); ++i) {
    fout << observations[i].first << '\t' << observations[i].second << endl;
    sprintf(buffer, "%s/depth%03d.jpg", path.c_str(), i);
    depths[i].saveImage(string(buffer));
  }

  for (auto i = 0; i < depths_threads.size(); ++i) {
    for (auto j = 0; j < depths_threads[i].size(); ++j) {
      sprintf(buffer, "%s/depth_t%d_f%03d.jpg", path.c_str(), i, j);
      depths_threads[i][j].saveImage(buffer);
    }
  }
  fout.close();
}

void fuseTwoSolution(CompactLabelSpace &s1, const CompactLabelSpace &s2,
                     const int pid, const MRFModel<int> *model) {
  //        CHECK_EQ(s1.getNumNode(), s2.getNumNode());
  //        CHECK_GT(s1.getNumNode(), 0);
  //        CHECK_LT(pid, s2.getLabelSpace()[0].size());

  const int kPix = s1.getNumNode();
  kolmogorov::qpbo::QPBO<int> qpbo(kPix, kPix * 2);
  qpbo.AddNode(kPix);
  for (auto i = 0; i < kPix; ++i)
    qpbo.AddUnaryTerm(i, model->operator()(i, s1(i, 0)),
                      model->operator()(i, s2(i, pid)));
  for (auto x = 0; x < model->width - 1; ++x) {
    for (auto y = 0; y < model->height - 1; ++y) {
      int e00, e01, e10, e11;
      int pix1 = y * model->width + x, pix2 = y * model->width + x + 1,
          pix3 = (y + 1) * model->width + x;
      int l10, l11, l12;
      l10 = s2(pix1, pid);
      l11 = s2(pix2, pid);
      l12 = s2(pix3, pid);
      // x direction
      e00 = model->computeSmoothCost(pix1, s1(pix1, 0), s1(pix2, 0), true);
      e01 = model->computeSmoothCost(pix1, s1(pix1, 0), l11, true);
      e10 = model->computeSmoothCost(pix1, l10, s1(pix2, 0), true);
      e11 = model->computeSmoothCost(pix1, l10, l11, true);
      qpbo.AddPairwiseTerm(pix1, pix2, e00, e01, e10, e11);

      // y direction
      e00 = model->computeSmoothCost(pix1, s1(pix1, 0), s1(pix3, 0), false);
      e01 = model->computeSmoothCost(pix1, s1(pix1, 0), l12, false);
      e10 = model->computeSmoothCost(pix1, l10, s1(pix3, 0), false);
      e11 = model->computeSmoothCost(pix1, l10, l12, false);
      qpbo.AddPairwiseTerm(pix1, pix3, e00, e01, e10, e11);
    }
  }

  qpbo.MergeParallelEdges();
  qpbo.Solve();
  qpbo.ComputeWeakPersistencies();

  for (auto i = 0; i < kPix; ++i) {
    if (qpbo.GetLabel(i) == 1)
      s1(i, 0) = s2(i, pid);
  }
}

void multiwayFusionByTRWS(const CompactLabelSpace &proposals,
                          const MRFModel<int> *model,
                          CompactLabelSpace &solution) {
  CHECK(!proposals.empty());
  const int singleLabelSize = (int)proposals.getSingleLabel().size();
  int fullLabelSize = 0;
  if (!proposals.getLabelSpace().empty())
    fullLabelSize += (int)proposals.getLabelSpace()[0].size();
  const int nLabel = singleLabelSize + fullLabelSize;

  const int &width = model->width;
  const int &height = model->height;

  typedef TypeGeneral SmoothT;
  typedef MRFEnergy<SmoothT> TRWS;

  const int kPix = model->width * model->height;
  TRWS::Options option;
  SmoothT::REAL energy, lowerBound;

  shared_ptr<TRWS> mrf(new TRWS(SmoothT::GlobalSize()));
  shared_ptr<TRWS::NodeId> nodes(new TRWS::NodeId[kPix]);

  // constructing energy function
  // data cost
  vector<vector<SmoothT::REAL>> dataCost((size_t)kPix);
  for (auto i = 0; i < kPix; ++i)
    dataCost[i].resize((size_t)nLabel, 0);

  for (auto i = 0; i < kPix; ++i) {
    for (auto l = 0; l < singleLabelSize; ++l)
      dataCost[i][l] = model->operator()(i, proposals[l]) / model->MRFRatio;
    for (auto l = 0; l < fullLabelSize; ++l)
      dataCost[i][l + singleLabelSize] =
          model->operator()(i, proposals(i, l)) / model->MRFRatio;
    nodes.get()[i] = mrf->AddNode(SmoothT::LocalSize(nLabel),
                                  SmoothT::NodeData(dataCost[i].data()));
  }

  // smoothness cost
  vector<vector<SmoothT::REAL>> smoothCost((size_t)2 * kPix);
  for (auto &sc : smoothCost)
    sc.resize((size_t)(nLabel * nLabel));

  auto computeSmoothCost = [&](const int pix1, bool direction) {
    const int pix2 = direction ? pix1 + 1 : pix1 + width;
    const int offset = direction ? 0 : kPix;
    for (int l1 = 0; l1 < nLabel; ++l1) {
      for (int l2 = 0; l2 < nLabel; ++l2) {
        int ll1, ll2;
        if (l1 < singleLabelSize)
          ll1 = proposals[l1];
        else
          ll1 = proposals(pix1, l1 - singleLabelSize);
        if (l2 < singleLabelSize)
          ll2 = proposals[l2];
        else
          ll2 = proposals(pix2, l2 - singleLabelSize);
        smoothCost[pix1 + offset][l1 + l2 * nLabel] =
            model->computeSmoothCost(pix1, ll1, ll2, direction) /
            model->MRFRatio;
      }
    }
  };

  for (auto y = 0; y < model->height - 1; ++y) {
    for (auto x = 0; x < model->width - 1; ++x) {
      const int pix1 = y * width + x;
      const int pix2 = y * width + x + 1;
      const int pix3 = (y + 1) * width + x;
      computeSmoothCost(pix1, true);
      computeSmoothCost(pix1, false);
      mrf->AddEdge(
          nodes.get()[pix1], nodes.get()[pix2],
          SmoothT::EdgeData(SmoothT::GENERAL, smoothCost[pix1].data()));
      mrf->AddEdge(
          nodes.get()[pix1], nodes.get()[pix3],
          SmoothT::EdgeData(SmoothT::GENERAL, smoothCost[pix1 + kPix].data()));
    }
  }

  // solve
  option.m_iterMax = 30;
  mrf->Minimize_TRW_S(option, lowerBound, energy);

  // printf("====================================\nTRWS final
  // energy:%.3f\n====================================\n", energy);
  // copy result
  if (solution.getNumNode() < kPix)
    solution.init(kPix, vector<int>(1, 0));
  for (auto i = 0; i < kPix; ++i) {
    const int l = mrf->GetSolution(nodes.get()[i]);
    if (l < singleLabelSize)
      solution(i, 0) = proposals[l];
    else
      solution(i, 0) = proposals(i, l - singleLabelSize);
  }
}

void dumpOutData(const ParallelFusionPipeline<CompactLabelSpace> &pipeline,
                 const string &prefix) {
  char buffer[1024] = {};
  const GlobalTimeEnergyProfile &profile = pipeline.getGlobalProfile();
  sprintf(buffer, "%s_global.txt", prefix.c_str());
  ofstream globalOut(buffer);
  CHECK(globalOut.is_open());
  for (const auto &ob : profile.getProfile())
    globalOut << ob.first << '\t' << ob.second << endl;
  globalOut.close();

  const vector<list<Observation>> &threadProfiles =
      pipeline.getAllThreadProfiles();
  for (auto tid = 0; tid < threadProfiles.size(); ++tid) {
    sprintf(buffer, "%s_thread%d.txt", prefix.c_str(), tid);
    ofstream threadOut(buffer);
    CHECK(threadOut.is_open());
    for (const auto &ob : threadProfiles[tid])
      threadOut << ob.first << '\t' << ob.second << endl;
    threadOut.close();
  }
}
} // namespace simple_stereo