#ifndef DISPATCH_LAYOUT_H
#define DISPATCH_LAYOUT_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "datacopy.h"
#include "sync_collectives.h"
#include "comm_group.h"

using namespace AscendC;
using namespace Moe;

template <typename T>
class DispatchLayout {
    constexpr uint32_t UB_32_ALIGN = 32U;

public:
    __aicore__ inline DispatchLayout() {};

    __aicore__ inline void Init(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert, GM_ADDR isTokenInRank, 
                                GM_ADDR workspace, Tpipe *pipe, const DispatchLayoutTilingData *tilingData)
    {
        topkIdxGM_.SetGlobalBuffer((__gm__ T*)topkIdx);
        numTokensPerRankGM_.SetGlobalBuffer((__gm__ T*)numTokensPerRank);
        numTokensPerExpertGM_.SetGlobalBuffer((__gm__ T*)numTokensPerExpert);
        isTokenInRankGM_.SetGlobalBuffer((__gm__ T*)isTokenInRank);

        numToken_ = tilingData->dispatchLayoutInfo.numToken;
        numRank_ = tilingData->dispatchLayoutInfo.numRank;
        numExpert_ = tilingData->dispatchLayoutInfo.numExpert;
        numTopk_ = tilingData->dispatchLayoutInfo.numTopk;

        topkIdx32AlignIntLen_ = Ceil(numToken_ * numTopk_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerRank32AlignIntLen_ = Ceil(numRank_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerExpert32AlignIntLen_ = Ceil(numExpert_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        isTokenInRank32AlignIntLen_ = Ceil(numToken_ * numRank_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx_ = GetBlockIdx();
        if (coreIdx_ != 0) {
            return;
        }
        tpipe_->Reset();
        tpipe_->InitBuffer(topkIdxBuf_, topkIdx32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerRankBuf_, numTokensPerRank32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
        tpipe_->InitBuffer(isTokenInRankBuf_, isTokenInRank32AlignIntLen_);
        tpipe_->InitBuffer(seenRankBuf_, numRank_ * sizeof(uint32_t));

        LocalTensor<uint32_t> topkIdxTensor = topkIdxBuf_.AllocTensor<uint32_t>();
        const DataCopyExtParams dataCopyParams{1U, topkIdx32AlignIntLen_, 0U, 0U, 0U};
        const DataCopyPadExtParams<SrcInfoType> padParams{false, 0U, 0U, 0U};
        DataCopyPad(topkIdxTensor, topkIdxGM_, dataCopyParams, padParams);

        LocalTensor<uint32_t> numTokensPerRankTensor = numTokensPerRankBuf_.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> isTokenInRankTensor = isTokenInRankBuf_.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> seenRankTensor = seenRankBuf_.AllocTensor<uint32_t>();
        Duplicate<uint32_t>(numTokensPerRankTensor, 0, numRank_);
        Duplicate<uint32_t>(numTokensPerExpertTensor, 0, numExpert_);
        Duplicate<uint32_t>(isTokenInRankTensor, 0, numToken_ * numRank_);

        int experts_per_rank = numExpert_ / numRank_;
        for (int i = 0; i < numToken_; ++i) {
            Duplicate<uint32_t>(seenRankTensor, 0, numRank_);
            for (int j = 0; j < numTopk_; ++j) {
                int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                if (expert_idx >= 0) {
                    numTokensPerExpertTensor.SetValue(expert_idx, numTokensPerExpertTensor.GetValue(expert_idx) + 1);
                    int rank_id = expert_idx / experts_per_rank;
                    if (!seenRankTensor.GetValue(rank_id)) {
                        numTokensPerRankTensor.SetValue(rank_id, numTokensPerRankTensor.GetValue(rank_id) + 1);
                        isTokenInRankTensor.SetValue(i * num_ranks + rank_id, 1);
                        seenRankTensor.SetValue(rank_id, 1);
                    }
                }
            }
        }

        const DataCopyExtParams numTokensPerRankDataCopyParams{1U, numTokensPerRank32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(numTokensPerRankGM_, numTokensPerRankTensor, numTokensPerRankDataCopyParams);
        const DataCopyExtParams numTokensPerExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(numTokensPerExpertGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
        const DataCopyExtParams isTokenInRankDataCopyParams{1U, isTokenInRank32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(isTokenInRankGM_, isTokenInRankTensor, isTokenInRankDataCopyParams);
    }

private:
    GlobalTensor<T> topkIdxGM_;
    GlobalTensor<T> numTokensPerRankGM_;
    GlobalTensor<T> numTokensPerExpertGM_;
    GlobalTensor<T> isTokenInRankGM_;

    TBuf<> topkIdxBuf_;
    TBuf<> numTokensPerRankBuf_;
    TBuf<> numTokensPerExpertBuf_;
    TBuf<> isTokenInRankBuf_;
    TBuf<> seenRankBuf_;

    uint32_t topkIdx32AlignIntLen_{0};
    uint32_t numTokensPerRank32AlignIntLen_{0};
    uint32_t numTokensPerExpert32AlignIntLen_{0};
    uint32_t isTokenInRank32AlignIntLen_{0};
};

#endif  // DISPATCH_LAYOUT_H
