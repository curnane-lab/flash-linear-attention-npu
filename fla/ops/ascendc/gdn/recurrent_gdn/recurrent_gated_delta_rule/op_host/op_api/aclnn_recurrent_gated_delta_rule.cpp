/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_recurrent_gated_delta_rule.cpp
 * \brief
 */
#include <dlfcn.h>
#include "aclnn_recurrent_gated_delta_rule.h"
#include "recurrent_gated_delta_rule.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t QUERY_DIM_NUM = 3;
constexpr size_t KEY_DIM_NUM = 3;
constexpr size_t VALUE_DIM_NUM = 3;
constexpr size_t BETA_DIM_NUM = 2;
constexpr size_t STATE_DIM_NUM = 4;
// Frozen v1 recover contract: the largest per-request window the kernel can
// write into records (RGDR_MAX_MTP in the tiling/kernel; a longer request
// makes the kernel return without writing).  records dim 2 must hold it --
// see CheckRecoverRecordsCapacity.
constexpr int64_t RECOVER_MAX_WINDOW = 8;

struct RecurrentGatedDeltaRuleParams {
    // madatory
    const aclTensor *query {nullptr};
    const aclTensor *key {nullptr};
    const aclTensor *value {nullptr};
    const aclTensor *beta {nullptr};
    const aclTensor *state {nullptr};
    const aclTensor *actual_seq_lengths {nullptr};
    const aclTensor *ssm_state_indices {nullptr};
    // optional
    const aclTensor *g {nullptr};
    const aclTensor *gk {nullptr};
    const aclTensor *num_accepted_tokens {nullptr};
    // attrs
    float scale {1.0f};
    //output
    const aclTensor *out {nullptr};
};

// support dtype
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16,op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BETA_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SSM_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ACC_TO_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};

static inline bool CheckNotNull(const RecurrentGatedDeltaRuleParams &params)
{
    // 必选参数
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.state, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.actual_seq_lengths, return false);
    OP_CHECK_NULL(params.ssm_state_indices, return false);
    OP_CHECK_NULL(params.out, return false);

    return true;
}

static inline bool CheckDtypeVaild(const RecurrentGatedDeltaRuleParams &params)
{
    // 检查必选参数数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST, return false);

    // 检查可选参数数据类型
    if (params.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    }
    if (params.gk != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.gk, G_TYPE_SUPPORT_LIST, return false);
    }
    if (params.num_accepted_tokens != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.num_accepted_tokens, ACC_TO_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(RecurrentGatedDeltaRuleParams &params)
{
    // 检查输入参数是否在支持的数据类型范围内
    CHECK_RET(CheckDtypeVaild(params), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("RecurrentGatedDeltaRule check params sucess.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(RecurrentGatedDeltaRuleParams &params)
{
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.beta->SetOriginalShape(params.beta->GetViewShape());
    params.state->SetOriginalShape(params.state->GetViewShape());
    params.actual_seq_lengths->SetOriginalShape(params.actual_seq_lengths->GetViewShape());
    params.ssm_state_indices->SetOriginalShape(params.ssm_state_indices->GetViewShape());

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
                                                         const aclTensor *value, const aclTensor *beta,
                                                         aclTensor *stateRef, const aclTensor *actualSeqLengths,
                                                         const aclTensor *ssmStateIndices, const aclTensor *g,
                                                         const aclTensor *gk, const aclTensor *numAcceptedTokens,
                                                         float scaleValue, aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRule,
                   DFX_IN(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
                          numAcceptedTokens, scaleValue),
                   DFX_OUT(stateRef, out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    RecurrentGatedDeltaRuleParams params {query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk, numAcceptedTokens,scaleValue, out};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    if (g != nullptr) {
        g = l0op::Contiguous(g, uniqueExecutor.get());
    }
    if (gk != nullptr) {
        gk = l0op::Contiguous(gk, uniqueExecutor.get());
    }
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
    }
    if (!IsContiguous(stateRef)) {
        stateRef = uniqueExecutor.get()->CreateView(
            stateRef,
            stateRef->GetViewShape(),
            stateRef->GetStorageShape(),
            stateRef->GetViewStrides(),
            stateRef->GetViewOffset());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());

    // 调用l0接口
    auto outRet =
        l0op::RecurrentGatedDeltaRule(query_, key_, value_, beta_, stateRef, actualSeqLengths_, ssmStateIndices_, g, gk,
                                      numAcceptedTokens, scaleValue, uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // 获取计算过程中需要使用的workspace大小。
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

namespace {
struct RecurrentGatedDeltaRuleRecoverParams {
    // madatory
    const aclTensor *query {nullptr};
    const aclTensor *key {nullptr};
    const aclTensor *value {nullptr};
    const aclTensor *beta {nullptr};
    const aclTensor *state {nullptr};
    const aclTensor *actual_seq_lengths {nullptr};
    const aclTensor *ssm_state_indices {nullptr};
    // optional
    const aclTensor *g {nullptr};
    const aclTensor *gk {nullptr};
    const aclTensor *num_accepted_tokens {nullptr};
    // recover extension (optional)
    const aclTensor *source_state {nullptr};
    const aclTensor *record_indices {nullptr};
    // attrs
    float scale {1.0f};
    //output
    const aclTensor *out {nullptr};
    const aclTensor *records {nullptr};
};

static inline bool CheckNotNullRecover(const RecurrentGatedDeltaRuleRecoverParams &params)
{
    // 必选参数
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.state, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.actual_seq_lengths, return false);
    OP_CHECK_NULL(params.ssm_state_indices, return false);
    OP_CHECK_NULL(params.out, return false);

    // record_indices 与 records 必须成对出现
    if ((params.record_indices == nullptr) != (params.records == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "record_indices and records must be provided together.");
        return false;
    }

    return true;
}

static inline bool CheckDtypeVaildRecover(const RecurrentGatedDeltaRuleRecoverParams &params)
{
    // 检查必选参数数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST, return false);

    // 检查可选参数数据类型
    if (params.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    }
    if (params.gk != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.gk, G_TYPE_SUPPORT_LIST, return false);
    }
    if (params.num_accepted_tokens != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.num_accepted_tokens, ACC_TO_TYPE_SUPPORT_LIST, return false);
    }
    if (params.source_state != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.source_state, STATE_TYPE_SUPPORT_LIST, return false);
    }
    if (params.record_indices != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.record_indices, SSM_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    if (params.records != nullptr) {
        const std::initializer_list<op::DataType> recordsSupport = {op::DataType::DT_FLOAT};
        OP_CHECK_DTYPE_NOT_SUPPORT(params.records, recordsSupport, return false);
    }
    return true;
}

static aclnnStatus PreProcessRecover(RecurrentGatedDeltaRuleRecoverParams &params)
{
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.beta->SetOriginalShape(params.beta->GetViewShape());
    params.state->SetOriginalShape(params.state->GetViewShape());
    params.actual_seq_lengths->SetOriginalShape(params.actual_seq_lengths->GetViewShape());
    params.ssm_state_indices->SetOriginalShape(params.ssm_state_indices->GetViewShape());
    if (params.source_state != nullptr) {
        params.source_state->SetOriginalShape(params.source_state->GetViewShape());
    }
    if (params.record_indices != nullptr) {
        params.record_indices->SetOriginalShape(params.record_indices->GetViewShape());
    }
    if (params.records != nullptr) {
        params.records->SetOriginalShape(params.records->GetViewShape());
    }

    return ACLNN_SUCCESS;
}

static inline bool CheckSourceStateLayoutRecover(const RecurrentGatedDeltaRuleRecoverParams &params)
{
    if (params.source_state == nullptr) {
        return true;
    }
    // The kernel reads source_state as contiguous V x K blocks (PrefetchState
    // copies a whole head row in one shot), so only the outer two dims may
    // carry a non-trivial stride.  A permuted inner pair (DV and DK are both
    // 128) keeps the 4-dim shape but would be read back as the wrong matrix,
    // so the layout is rejected here rather than materialised or misread.
    // The tiling layer repeats this check for callers that bypass aclnn.
    const auto &strides = params.source_state->GetViewStrides();
    if (strides.size() != STATE_DIM_NUM) {
        // Rank mismatches are reported by the tiling shape check.
        return true;
    }
    const int64_t dk = params.state->GetViewShape().GetDim(STATE_DIM_NUM - 1);
    if (strides[STATE_DIM_NUM - 1] != 1 || strides[STATE_DIM_NUM - 2] != dk) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "source_state last two dims must be contiguous (stride(-1)==1, stride(-2)==DK=%ld), "
                "got stride(-1)=%ld stride(-2)=%ld",
                static_cast<long>(dk), static_cast<long>(strides[STATE_DIM_NUM - 1]),
                static_cast<long>(strides[STATE_DIM_NUM - 2]));
        return false;
    }
    return true;
}

static inline bool CheckRecoverRecordsCapacity(const RecurrentGatedDeltaRuleRecoverParams &params)
{
    if (params.records == nullptr) {
        return true;
    }
    // The kernel writes local steps [0, seqLen) of each request at
    // records[.., .., step, ..].  The host cannot read the per-request lengths
    // (device tensor values are the caller/adapter's responsibility, no D2H),
    // so the formal entry freezes the step capacity at the maximum window a
    // launch can ever write (RECOVER_MAX_WINDOW == the kernel's MAX_MTP): a
    // smaller dim 2 lets a step write spill into the next head's row, and past
    // the last head out of bounds.  The tiling layer repeats this check for
    // callers that bypass aclnn.
    const auto &recShape = params.records->GetViewShape();
    if (recShape.GetDimNum() != STATE_DIM_NUM) {
        // Rank mismatches are reported by the tiling shape check.
        return true;
    }
    if (recShape.GetDim(2) < RECOVER_MAX_WINDOW) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "records dim 2 (per-request step capacity) must be >= MAX_MTP=%ld, got %ld",
                static_cast<long>(RECOVER_MAX_WINDOW), static_cast<long>(recShape.GetDim(2)));
        return false;
    }
    return true;
}
} // namespace

aclnnStatus aclnnRecurrentGatedDeltaRuleRecoverGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    const aclTensor *sourceState, aclTensor *stateRef, const aclTensor *actualSeqLengths,
    const aclTensor *ssmStateIndices, const aclTensor *recordIndices, const aclTensor *g, const aclTensor *gk,
    const aclTensor *numAcceptedTokens, float scaleValue, aclTensor *out, aclTensor *records, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRuleRecover,
                   DFX_IN(query, key, value, beta, sourceState, stateRef, actualSeqLengths, ssmStateIndices,
                          recordIndices, g, gk, numAcceptedTokens, scaleValue),
                   DFX_OUT(stateRef, out, records));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    RecurrentGatedDeltaRuleRecoverParams params {query,        key,           value,   beta,         stateRef,
                                                 actualSeqLengths, ssmStateIndices, g, gk,          numAcceptedTokens,
                                                 sourceState, recordIndices, scaleValue, out, records};

    CHECK_RET(CheckNotNullRecover(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeVaildRecover(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSourceStateLayoutRecover(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRecoverRecordsCapacity(params), ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcessRecover(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    if (g != nullptr) {
        g = l0op::Contiguous(g, uniqueExecutor.get());
    }
    if (gk != nullptr) {
        gk = l0op::Contiguous(gk, uniqueExecutor.get());
    }
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
    }
    if (recordIndices != nullptr) {
        recordIndices = l0op::Contiguous(recordIndices, uniqueExecutor.get());
    }
    // source_state is declared IgnoreContiguous like state: keep its view so the
    // host tiling sees its strides (the last-two-dims contiguity contract is
    // enforced above in CheckSourceStateLayoutRecover and again in the tiling).
    if (sourceState != nullptr && !IsContiguous(sourceState)) {
        sourceState = uniqueExecutor.get()->CreateView(
            sourceState,
            sourceState->GetViewShape(),
            sourceState->GetStorageShape(),
            sourceState->GetViewStrides(),
            sourceState->GetViewOffset());
    }
    if (!IsContiguous(stateRef)) {
        stateRef = uniqueExecutor.get()->CreateView(
            stateRef,
            stateRef->GetViewShape(),
            stateRef->GetStorageShape(),
            stateRef->GetViewStrides(),
            stateRef->GetViewOffset());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());

    // 调用l0接口
    auto outRet = l0op::RecurrentGatedDeltaRuleRecover(query_, key_, value_, beta_, sourceState, stateRef,
                                                       actualSeqLengths_, ssmStateIndices_, recordIndices, g, gk,
                                                       numAcceptedTokens, scaleValue, records, uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // 获取计算过程中需要使用的workspace大小。
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentGatedDeltaRuleRecover(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRuleRecover);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif