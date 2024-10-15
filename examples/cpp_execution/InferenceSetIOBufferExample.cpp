//-----------------------------------------------------------------------------
//
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
//-----------------------------------------------------------------------------

#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "/opt/qti-aic/dev/inc/QAicApi.hpp"
#include "/opt/qti-aic/dev/inc/qaicapihpp/QAicApiDataTypes.hpp"

namespace py = pybind11;

namespace
{

    /**
     * Simple helper to return true if the buffer mapping instance is an input one
     * @param bufmap buffer mapping instance
     * @return true if the instance is an input buffer one.
     */
    [[nodiscard]] bool isInputBuffer(const qaic::rt::BufferMapping &bufmap)
    {
        return bufmap.ioType == BUFFER_IO_TYPE_INPUT;
    }

    /**
     * Populate input, output vectors with QBuffer information
     * @param bufmap Buffer mapping instance
     * @param buf Actual QBufffer that was generated at callsite/caller.
     * @param inputBuffers Vector to use in case this is input instance
     * @param outputBuffers Vector to use in case this is an output instance
     */
    void populateVector(const qaic::rt::BufferMapping &bufmap, const QBuffer &buf,
                        std::vector<QBuffer> &inputBuffers,
                        std::vector<QBuffer> &outputBuffers)
    {

        if (isInputBuffer(bufmap))
        {
            {
                inputBuffers.push_back(buf);
            }
        }
        else
        {
            outputBuffers.push_back(buf);
        }
    }

    class QBufferWrapper
    {
    public:
        explicit QBufferWrapper(size_t size) : buffer_{size, new uint8_t[size]} {}
        ~QBufferWrapper() { delete[] buffer_.buf; }

        [[nodiscard]] QBuffer &getQBuffer() { return buffer_; }

    private:
        QBuffer buffer_;
    };
    using shQBufferWrapper = std::shared_ptr<QBufferWrapper>;

    [[nodiscard]] shQBufferWrapper
    createBuffer(const std::string &bufferName,
                 const qaic::rt::BufferMappings &allBufferMappings)
    {
        auto it =
            std::find_if(allBufferMappings.begin(), allBufferMappings.end(),
                         [&bufferName](const qaic::rt::BufferMapping &bufferMapping)
                         {
                             return (bufferName == bufferMapping.bufferName);
                         });
        if (it != allBufferMappings.end())
        {
            return std::make_shared<QBufferWrapper>(it->size);
        }

        throw std::runtime_error(
            "Buffer mapping of Input Type not found for buffer named : " +
            bufferName);
    }

    [[nodiscard]] shQBufferWrapper
    createDecodeBuffer(const std::string &bufferName,
                       const qaic::rt::BufferMappings &allBufferMappings)
    {
        auto it =
            std::find_if(allBufferMappings.begin(), allBufferMappings.end(),
                         [&bufferName](const qaic::rt::BufferMapping &bufferMapping)
                         {
                             return (bufferName == bufferMapping.bufferName);
                         });
        if (it != allBufferMappings.end())
        {
            return std::make_shared<QBufferWrapper>(1);
        }

        throw std::runtime_error(
            "Buffer mapping of Decode Input Type not found for buffer named : " +
            bufferName);
    }

    /**
     * Populate input, output vectors with QBuffer information
     * @param outputBuffers Vector to use in case this is an output instance
     * @param nextTokenIds Vector to store output[logits]
     */
    std::vector<int64_t> get_logits_from_output_buffers(
        std::vector<QBuffer> &outputBuffers,
        std::vector<int64_t> &nextTokenIds)
    {
        auto rawOPBufPtr = outputBuffers[outputBuffers.size() - 1].buf;
        const float *bufferOT = reinterpret_cast<const float *>(rawOPBufPtr);
        int size_of_logits = outputBuffers[outputBuffers.size() - 1].size / sizeof(float);

        // Calculate the index of the maximum element
        auto maxElementIter = std::max_element(bufferOT, bufferOT + size_of_logits);
        int maxElementIndex = std::distance(bufferOT, maxElementIter);

        nextTokenIds.push_back(maxElementIndex);
        std::vector<int64_t> logits({maxElementIndex});
        return logits;
    }

    /**
     * Given a Input buffer and size, populate it with inputs data(input_ids or position ids)
     * @param inputBuffer buffer to populate
     * @param tokenVector vector to fill input Buffer
     */
    void populateBuffer(QBuffer &inputBuffer,
                        const std::vector<int64_t> &tokenVector)
    {
        size_t token_size_bytes = tokenVector.size() * sizeof(int64_t);
        if (inputBuffer.size < token_size_bytes)
        {
            delete[] inputBuffer.buf;
            inputBuffer.buf = new uint8_t[token_size_bytes];
            inputBuffer.size = token_size_bytes;
        }
        std::copy_n(reinterpret_cast<const uint8_t *>(tokenVector.data()),
                    token_size_bytes, inputBuffer.buf);

        if (inputBuffer.size > token_size_bytes)
        {
            std::memset(inputBuffer.buf + token_size_bytes, 0,
                        inputBuffer.size - token_size_bytes);
        }
    }

    template <typename T>
    [[nodiscard]] std::string qBufferToString(shQBufferWrapper wrappedBuf)
    {
        std::ostringstream strm;
        auto rawBufPtr = wrappedBuf->getQBuffer().buf;
        const T *bufferT = reinterpret_cast<const T *>(rawBufPtr);
        int numT = wrappedBuf->getQBuffer().size / sizeof(T);
        for (int i = 0; i < numT; i++)
        {
            strm << "[ " << i << " ] = " << bufferT[i] << "\n";
        }
        return strm.str();
    }

    /**
     * Given buffer mapping instance, return true if this instance does not
     * contain input or output buffers (e.g. it contains uninitialized or invalid)
     * @param bufmap buffer mapping instance
     * @return true if the buffer mapping instance does not container a valid buffer
     */
    [[nodiscard]] bool notInputOrOutput(const qaic::rt::BufferMapping &bufmap)
    {
        const std::initializer_list<QAicBufferIoTypeEnum> bufTypes{
            BUFFER_IO_TYPE_INPUT, BUFFER_IO_TYPE_OUTPUT};
        const auto func([type = bufmap.ioType](const auto v)
                        { return v == type; });
        return std::none_of(bufTypes.begin(), bufTypes.end(), func);
    }

    /**
     * Given input and output buffers, release all heap allocated
     * @param bufferMappings vector of BufferMapping
     * @param inputBuffers vector of QBuffers - inputs
     * @param outputBuffers vector of Qbuffers - outputs
     * @param inputIdBuffers Qbuffers - input id
     * @param positionIdBuffers Qbuffers - position id
     */
    void populateBuffersWithInputs(const std::vector<qaic::rt::BufferMapping> bufferMappings,
                                   std::vector<QBuffer> &inputBuffers,
                                   std::vector<QBuffer> &outputBuffers,
                                   QBuffer &inputIdBuffer,
                                   QBuffer &positionIdBuffer)
    {
        inputBuffers.clear();
        outputBuffers.clear();
        for (const auto &bufmap : bufferMappings)
        {
            if (notInputOrOutput(bufmap)) {
                continue;
            }
            QBuffer buf{bufmap.size, new uint8_t[bufmap.size]};
            populateVector(bufmap, buf, inputBuffers, outputBuffers);
        }
        // Filling last 2 index of inputBuffers with inputIds and positionIds
        inputBuffers[inputBuffers.size() - 1] = positionIdBuffer;
        inputBuffers[inputBuffers.size() - 2] = inputIdBuffer;
    }
} // namespace

int generatePrompt(
    py::object tokenizer,
    const std::string &qpcPath,
    int batch_size,
    int ctx_len,
    std::optional<std::vector<std::string>> prompt = std::nullopt, // prompt_len
    std::optional<int> generation_len = std::nullopt,
    std::optional<std::vector<int>> device_id = std::nullopt)
{
    try
    {
        py::module sys = py::module::import("sys");
        // Add own path for examples folder
        sys.attr("path").attr("append")("examples/cpp_execution");
        
        py::module text_generation_inference = py::module::import("text_inference_using_cpp");

        // QID Generation
        std::vector<QID> qidList;
        if (device_id.has_value())
        {
            for (const auto &id : device_id.value())
            {
                try
                {
                    int32_t qid = id;
                    qidList.push_back(qid);
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "Invalid device id string" << std::endl;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "Device id string " << id << " is out of range!" << std::endl;
                }
            }
        }
        else
        {
            // need to use auto device picker
            qidList.push_back(1);
        }

        // *** CONTEXT ***
        constexpr QAicContextProperties_t *NullProp = nullptr;
        auto context = qaic::rt::Context::Factory(NullProp, qidList); // session == context

        // *** QPC ***
        auto qpc = qaic::rt::Qpc::Factory(qpcPath);

        // TODO: prefill_seq_len  from context
        int prefill_seq_len = 32;

        // Get input dict from python
        py::dict inputs = text_generation_inference.attr("tokenize_for_prefill")(prompt, tokenizer, 32).cast<py::dict>();

        py::array input_ids_py = inputs["input_ids"].cast<py::array>();
        py::buffer_info inp_id_buf = input_ids_py.request();
        std::vector<int64_t> token_input_ids;
        int64_t *input_id_ptr = static_cast<int64_t *>(inp_id_buf.ptr);
        for (ssize_t i = 0; i < inp_id_buf.shape[0]; ++i)
        {
            for (ssize_t j = 0; j < inp_id_buf.shape[1]; ++j)
            {
                token_input_ids.push_back(input_id_ptr[i * (inp_id_buf.shape[1]) + j]);
            }
        }

        py::array attention_mask_py = inputs["attention_mask"].cast<py::array>();
        auto attn_mask_buff = attention_mask_py.request();
        int64_t *attn_mask_ptr = static_cast<int64_t *>(attn_mask_buff.ptr);
        std::vector<int64_t> attention_mask;

        for (ssize_t i = 0; i < attn_mask_buff.shape[0]; ++i)
        {
            for (ssize_t j = 0; j < attn_mask_buff.shape[1]; j++)
            {
                attention_mask.push_back(attn_mask_ptr[i * (attn_mask_buff.shape[1]) + j]);
            }
        }
        ssize_t padded_len = token_input_ids.size();
        int num_chunks = -(padded_len / -prefill_seq_len);
        padded_len = num_chunks * prefill_seq_len;

        // assert(stream == false);
        // assert(generation_len.value() > 0); // TODO: Enable when getting values from Py file

        // PREPARE INPUTS FOR PREFILL
        std::vector<u_int64_t> arrange_vector(padded_len); // Initialize with -1
        for (int i = 0; i < (int)arrange_vector.size(); ++i)
        {
            arrange_vector[i] = i;
        }

        // Create position_ids vector
        std::vector<int64_t> position_ids(padded_len);

        for (int64_t i = 0; i < padded_len; ++i)
        {
            position_ids[i] = attn_mask_ptr[i] ? arrange_vector[i] : -1;
        }

        auto max_input_len_value = std::max_element(position_ids.begin(), position_ids.end());
        if (!generation_len.has_value())
        {
            generation_len = ctx_len - std::accumulate(attention_mask.begin(), attention_mask.end(), 0);
        }

        //*** RUN PREFILL *** //TODO: Adding chunks
        const auto &bufferMappings = qpc->getBufferMappings();
        const auto &bufferMappings2 = qpc->getBufferMappingsV2();

        auto inputIdBuffer = createBuffer("input_ids", bufferMappings);
        populateBuffer(inputIdBuffer->getQBuffer(), token_input_ids);

        auto positionIdBuffer = createBuffer("position_ids", bufferMappings);
        populateBuffer(positionIdBuffer->getQBuffer(), position_ids);

        // *** INFERENCE SET ***
        constexpr uint32_t setSize = 1;
        constexpr uint32_t numActivations = 1;
        auto inferenceSet = qaic::rt::InferenceSet::Factory(
            context, qpc, qidList.at(0), setSize, numActivations);

        // *** SETUP IO BUFFERS ***
        qaic::rt::shInferenceHandle submitHandle;
        auto status = inferenceSet->getAvailable(submitHandle);
        if (status != QS_SUCCESS)
        {
            std::cerr << "Error obtaining Inference Handle\n";
            return -1;
        }

        std::vector<QBuffer> inputBuffers;
        std::vector<QBuffer> outputBuffers;

        populateBuffersWithInputs(bufferMappings,
                                  inputBuffers,
                                  outputBuffers,
                                  inputIdBuffer->getQBuffer(),
                                  positionIdBuffer->getQBuffer());

        submitHandle->setInputBuffers(inputBuffers);
        submitHandle->setOutputBuffers(outputBuffers);

        qaic::rt::BufferIdentifiers bufferIdentifiers(bufferMappings2);
        std::vector<std::pair<uint32_t, std::vector<uint32_t>>> bufDim = bufferIdentifiers.getBufferSizeDimensionPair();

        for (auto &bufid : bufferIdentifiers.getBufferIdentifierVec())
        {
            if (bufid.getBufferName().find("past_") == 0)
            {
                bufDim[bufid.getBufferIndex()].second = std::vector{0U};
            }
        }

        submitHandle->setBufferDimensions(bufDim);

        // *** SUBMIT ***
        constexpr uint32_t inferenceId = 0; // also named as request ID
        status = inferenceSet->submit(submitHandle, inferenceId);
        if (status != QS_SUCCESS)
        {
            std::cerr << "Error in submitting handle through InferenceSet\n";
            return -1;
        }

        // *** COMPLETION ***
        qaic::rt::shInferenceHandle completedHandle;
        status = inferenceSet->getCompletedId(completedHandle, inferenceId);
        if (status != QS_SUCCESS)
        {
            std::cerr << "Error in getting completed handle through InferenceSet\n";
            return -1;
        }
        status = inferenceSet->putCompleted(std::move(completedHandle));
        if (status != QS_SUCCESS)
        {
            std::cerr << "Error in putting completed handle through InferenceSet\n";
            return -1;
        }

        // *** GET OUTPUT ***
        //
        // At this point, the output is available in "outputBuffers" and can be
        // consumed.
        //

        // DECODE LOOP
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int64_t>> generated_ids;

        for (int bs = 0; bs < batch_size; bs++)
        {
            std::vector<int64_t> nextTokenIds;
            for (int num_tokens = 1; num_tokens < generation_len; num_tokens++)
            {
                std::vector<int64_t> logits = get_logits_from_output_buffers(outputBuffers, nextTokenIds);
                std::vector<int64_t> position_id_for_decode({*max_input_len_value + num_tokens});

                // Making past_ values Null
                qaic::rt::BufferIdentifiers bufferIdentifiers(bufferMappings2);
                std::vector<std::pair<uint32_t, std::vector<uint32_t>>> bufDim = bufferIdentifiers.getBufferSizeDimensionPair();

                for (auto &bufid : bufferIdentifiers.getBufferIdentifierVec())
                {
                    if (bufid.getBufferName().find("past_") == 0)
                    {
                        bufDim[bufid.getBufferIndex()].second = std::vector{0U};
                    }
                    // Making dim of input buffer to be (1,1)
                    if (bufid.getBufferName().find("input_ids") == 0)
                    {
                        int size = bufDim[bufid.getBufferIndex()].second.size();
                        bufDim[bufid.getBufferIndex()].first = 8;
                        for (int i = 0; i < size; i++)
                        {
                            bufDim[bufid.getBufferIndex()].second[i] = 1;
                        }
                    }
                    if (bufid.getBufferName().find("position_ids") == 0)
                    {
                        int size = bufDim[bufid.getBufferIndex()].second.size();
                        bufDim[bufid.getBufferIndex()].first = 8;
                        for (int i = 0; i < size; i++)
                            bufDim[bufid.getBufferIndex()].second[i] = 1;
                    }
                }
                submitHandle->setBufferDimensions(bufDim);

                auto inputIdBufferDecode = createDecodeBuffer("input_ids", bufferMappings);
                populateBuffer(inputIdBufferDecode->getQBuffer(), logits); // change it to logits
                auto positionIdBufferDecode = createDecodeBuffer("position_ids", bufferMappings);
                populateBuffer(positionIdBufferDecode->getQBuffer(), position_id_for_decode);

                populateBuffersWithInputs(bufferMappings,
                                          inputBuffers,
                                          outputBuffers,
                                          inputIdBufferDecode->getQBuffer(),
                                          positionIdBufferDecode->getQBuffer());
                submitHandle->setInputBuffers(inputBuffers);
                submitHandle->setOutputBuffers(outputBuffers);

                // *** SUBMIT ***
                constexpr uint32_t inferenceId = 0; // also named as request ID
                status = inferenceSet->submit(submitHandle, inferenceId);
                if (status != QS_SUCCESS)
                {
                    std::cerr << "Error in submitting handle through InferenceSet\n";
                    return -1;
                }
                // *** COMPLETION ***
                qaic::rt::shInferenceHandle completedHandle;
                status = inferenceSet->getCompletedId(completedHandle, inferenceId);
                if (status != QS_SUCCESS)
                {
                    std::cerr << "Error in getting completed handle through InferenceSet\n";
                    return -1;
                }
                status = inferenceSet->putCompleted(std::move(completedHandle));
                if (status != QS_SUCCESS)
                {
                    std::cerr << "Error in putting completed handle through InferenceSet\n";
                    return -1;
                }
            }
            get_logits_from_output_buffers(outputBuffers, nextTokenIds); // For last token id
            generated_ids.push_back(nextTokenIds);
            nextTokenIds.clear();
        }

        // *** Release user allocated buffers ***
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Decode Elapsed time: " << elapsed.count() << " seconds\n";

        // Sending Generated Ids to Python to Generated Text using Tokenizer
        text_generation_inference.attr("tokenize_decode_output")(tokenizer, generated_ids).cast<py::array>();
    }

    catch (const py::error_already_set &e)
    {
        std::cerr << "Python error: " << e.what() << std::endl;
    }
    return 0;
}

PYBIND11_MODULE(InferenceSetIOBufferExample, m)
{
    m.doc() = "Running PyBind11";

    m.def("generatePrompt", &generatePrompt, "generatePrompt function");
}