/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_STRUCT_H
#define RECURRENT_GATED_DELTA_RULE_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace RecurrentGatedDeltaRule {

#define RGDR_TPL_BF16 10
#define RGDR_TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(RecurrentGatedDeltaRule,
    ASCENDC_TPL_DTYPE_DECL(D_T_STATE, RGDR_TPL_BF16, RGDR_TPL_FP32),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_STATE, RGDR_TPL_BF16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_STATE, RGDR_TPL_FP32),
    ),
);
#endif

#pragma pack(push, 8)
struct alignas(8) RecurrentGatedDeltaRuleTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
    uint32_t stateStride0;
 	uint32_t stateStride1;
 	uint32_t stateStride2;
    // N2 extension: appended tail keeps every legacy offset below unchanged, so a
    // tiling buffer produced by the old host tiling function still parses, and a
    // legacy call site (new inputs absent) still selects the legacy path.
    uint32_t hasSourceState;       // 1 => read the initial state from `source_state`
    uint32_t hasRecords;           // 1 => write per-token records into `records`
    uint32_t recordStride0;        // records row stride       [elements]
    uint32_t recordStride1;        // records head stride      [elements]
    uint32_t recordStride2;        // records step stride      [elements]
    uint32_t sourceStateStride0;   // source_state row stride  [elements]
    uint32_t sourceStateStride1;   // source_state head stride [elements]
    uint32_t sourceStateStride2;   // source_state v stride    [elements]
    // Inner pack selector for one records row.  0 = the frozen P1 packing
    //   [0,V) = dc | [V,V+K) = normalized key | [V+K] = alpha
    // Any other value is an unsupported layout: the kernel then leaves records
    // untouched and the host layer is responsible for rejecting the call.
    uint32_t recordPackMode;
};
#pragma pack(pop)

} // namespace RecurrentGatedDeltaRule

#endif // RECURRENT_GATED_DELTA_RULE_STRUCT_H
