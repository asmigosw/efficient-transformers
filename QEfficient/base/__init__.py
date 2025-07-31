# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.base.common import QEFFCommonLoader  # noqa: F401
# from QEfficient.transformers.models.modeling_auto import (  # noqa: F401
#     QEFFAutoModel,
#     QEFFAutoModelForCausalLM,
#     QEFFAutoModelForImageTextToText,
#     QEFFAutoModelForSpeechSeq2Seq,
# )


# __init__.py
def get_qeff_models():
    from QEfficient.transformers.models.modeling_auto import (
        QEFFAutoModel,
        QEFFAutoModelForCausalLM,
        QEFFAutoModelForImageTextToText,
        QEFFAutoModelForSpeechSeq2Seq,
    )
    return {
        "QEFFAutoModel": QEFFAutoModel,
        "QEFFAutoModelForCausalLM": QEFFAutoModelForCausalLM,
        "QEFFAutoModelForImageTextToText": QEFFAutoModelForImageTextToText,
        "QEFFAutoModelForSpeechSeq2Seq": QEFFAutoModelForSpeechSeq2Seq,
    }