# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG GOVERSION=1.24

FROM golang:$GOVERSION AS builder

ARG TARGETPLATFORM

WORKDIR /workspace

# install libopus and ONNX Runtime for DeepFilterNet2
RUN apt-get update && apt-get install -y pkg-config libopus-dev libopusfile-dev libsoxr-dev wget

# Install ONNX Runtime for DeepFilterNet2 support
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.22.0.tgz && \
    mkdir -p /usr/local/include/onnxruntime && \
    cp onnxruntime-linux-x64-1.22.0/include/*.h /usr/local/include/onnxruntime/ && \
    cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so* /usr/local/lib/ && \
    ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/libonnxruntime.so && \
    ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/onnxruntime.so && \
    ldconfig && \
    ls -la /usr/local/lib/*onnx* && \
    rm -rf onnxruntime-linux-x64-1.22.0*

# download go modules
COPY go.mod .
COPY go.sum .
RUN go mod download

# copy source
COPY res/ res/
COPY cmd/ cmd/
COPY pkg/ pkg/
COPY version/ version/
COPY models/ models/

# Verify models were copied in builder stage
RUN ls -la models/ || echo "❌ models directory not found in builder"
RUN ls -la models/deepfilternet2/ || echo "❌ deepfilternet2 directory not found in builder"

# build
# Set CGO environment variables for ONNX Runtime
ENV CGO_CFLAGS="-I/usr/local/include/onnxruntime"
ENV CGO_LDFLAGS="-L/usr/local/lib -lonnxruntime"

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then GOARCH=arm64; else GOARCH=amd64; fi && \
    CGO_ENABLED=1 GOOS=linux GOARCH=${GOARCH} GO111MODULE=on go build -a -o livekit-sip ./cmd/livekit-sip

FROM golang:$GOVERSION

# install runtime dependencies for health check and ONNX Runtime
RUN apt-get update && \
    apt-get install -y libopus0 libopusfile0 libsoxr0 wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime runtime libraries
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.22.0.tgz && \
    ls -la onnxruntime-linux-x64-1.22.0/lib/ && \
    cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so* /usr/local/lib/ && \
    ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/libonnxruntime.so && \
    ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/onnxruntime.so && \
    ldconfig -v && \
    ls -la /usr/local/lib/*onnx* && \
    rm -rf onnxruntime-linux-x64-1.22.0*

# copy binary
COPY --from=builder /workspace/livekit-sip /bin/
# copy DeepFilterNet2 models
COPY --from=builder /workspace/models/ /sip/models/

# Verify models were copied to final stage
RUN ls -la /sip/models/ || echo "❌ /sip/models directory not found in final stage"
RUN ls -la /sip/models/deepfilternet2/ || echo "❌ /sip/models/deepfilternet2 directory not found in final stage"

# Set runtime environment variables for ONNX Runtime
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /sip

# Verify ONNX Runtime library is available
RUN ldd /bin/livekit-sip | grep onnx || echo "ONNX Runtime not linked (may be loaded dynamically)"
RUN ldconfig -p | grep onnx || echo "ONNX Runtime library paths:"
RUN ls -la /usr/local/lib/*onnx* || echo "ONNX Runtime libraries not found in /usr/local/lib"

# run
ENTRYPOINT ["livekit-sip", "--config=/sip-config/config.yaml"]
