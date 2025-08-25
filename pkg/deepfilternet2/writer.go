// Copyright 2023 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package deepfilternet2

import (
	"fmt"

	msdk "github.com/livekit/media-sdk"
	"github.com/livekit/protocol/logger"
)

// PCM16Writer wraps an existing PCM16Writer and applies DeepFilterNet2 noise suppression
type PCM16Writer struct {
	next      msdk.Writer[msdk.PCM16Sample]
	processor *Processor
	log       logger.Logger

	// Sample rate conversion buffers
	sampleRate      int
	needsResampling bool
	resampleBuffer  []float32
	resampleTempIn  []float32
	resampleTempOut []float32
}

// NewPCM16Writer creates a new DeepFilterNet2-enabled PCM16Writer
func NewPCM16Writer(next msdk.Writer[msdk.PCM16Sample], modelsPath string, log logger.Logger) (*PCM16Writer, error) {
	log.Infow("creating DeepFilterNet2 PCM16Writer", "modelsPath", modelsPath, "nextWriter", next.String())

	processor, err := NewProcessor(modelsPath, log.WithValues("component", "processor"))
	if err != nil {
		return nil, fmt.Errorf("failed to create DeepFilterNet2 processor: %w", err)
	}

	sampleRate := next.SampleRate()
	needsResampling := sampleRate != SampleRate

	w := &PCM16Writer{
		next:            next,
		processor:       processor,
		log:             log.WithValues("component", "writer"),
		sampleRate:      sampleRate,
		needsResampling: needsResampling,
	}

	if needsResampling {
		log.Infow("sample rate conversion required",
			"inputSampleRate", SampleRate,
			"outputSampleRate", sampleRate)
		// Initialize resampling buffers
		w.resampleBuffer = make([]float32, HopSize*2)
		w.resampleTempIn = make([]float32, HopSize)
		w.resampleTempOut = make([]float32, HopSize*sampleRate/SampleRate)
	}

	log.Infow("DeepFilterNet2 PCM16Writer created successfully",
		"sampleRate", sampleRate,
		"needsResampling", needsResampling)

	return w, nil
}

// WriteSample processes audio samples through DeepFilterNet2 and forwards to the next writer
func (w *PCM16Writer) WriteSample(samples msdk.PCM16Sample) error {
	if len(samples) == 0 {
		return nil
	}

	w.log.Debugw("processing samples through DeepFilterNet2",
		"inputSamples", len(samples),
		"sampleRate", w.sampleRate)

	// Convert int16 to float32 for processing
	floatSamples := w.int16ToFloat32(samples)

	// Apply sample rate conversion if needed
	var processedFloat []float32
	if w.needsResampling {
		// Upsample to 48kHz for DeepFilterNet2
		upsampled := w.upsample(floatSamples, w.sampleRate, SampleRate)

		// Process through DeepFilterNet2 at 48kHz
		filtered := w.processor.ProcessAudio(upsampled)

		// Downsample back to original sample rate
		processedFloat = w.downsample(filtered, SampleRate, w.sampleRate)
	} else {
		// Direct processing at 48kHz
		processedFloat = w.processor.ProcessAudio(floatSamples)
	}

	// Convert back to int16
	processedSamples := w.float32ToInt16(processedFloat)

	w.log.Debugw("completed DeepFilterNet2 processing",
		"outputSamples", len(processedSamples),
		"compressionRatio", float64(len(samples))/float64(len(processedSamples)))

	// Forward to next writer
	return w.next.WriteSample(processedSamples)
}

// String returns a string representation of the writer
func (w *PCM16Writer) String() string {
	return fmt.Sprintf("DeepFilterNet2Writer(%d) -> %s", w.sampleRate, w.next.String())
}

// SampleRate returns the sample rate of the writer
func (w *PCM16Writer) SampleRate() int {
	return w.sampleRate
}

// Close cleans up resources
func (w *PCM16Writer) Close() error {
	w.log.Debugw("closing DeepFilterNet2 PCM16Writer")

	if w.processor != nil {
		w.processor.Close()
	}

	// Don't close the next writer - that's managed by the caller
	w.log.Infow("DeepFilterNet2 PCM16Writer closed")
	return nil
}

// int16ToFloat32 converts int16 samples to float32 normalized to [-1, 1]
func (w *PCM16Writer) int16ToFloat32(samples msdk.PCM16Sample) []float32 {
	floatSamples := make([]float32, len(samples))
	for i, sample := range samples {
		floatSamples[i] = float32(sample) / 32768.0
	}
	return floatSamples
}

// float32ToInt16 converts float32 samples to int16 with clamping
func (w *PCM16Writer) float32ToInt16(samples []float32) msdk.PCM16Sample {
	int16Samples := make(msdk.PCM16Sample, len(samples))
	for i, sample := range samples {
		// Clamp to valid range
		if sample > 1.0 {
			sample = 1.0
		} else if sample < -1.0 {
			sample = -1.0
		}
		int16Samples[i] = int16(sample * 32767.0)
	}
	return int16Samples
}

// upsample performs simple linear interpolation upsampling
// This is a simplified resampler - production code might use a more sophisticated algorithm
func (w *PCM16Writer) upsample(input []float32, inputRate, outputRate int) []float32 {
	if inputRate == outputRate {
		return input
	}

	ratio := float64(outputRate) / float64(inputRate)
	outputLen := int(float64(len(input)) * ratio)
	output := make([]float32, outputLen)

	w.log.Debugw("upsampling audio",
		"inputRate", inputRate,
		"outputRate", outputRate,
		"ratio", ratio,
		"inputLen", len(input),
		"outputLen", outputLen)

	for i := range output {
		srcIndex := float64(i) / ratio
		srcIndexInt := int(srcIndex)
		fraction := srcIndex - float64(srcIndexInt)

		if srcIndexInt >= len(input)-1 {
			output[i] = input[len(input)-1]
		} else {
			// Linear interpolation
			output[i] = input[srcIndexInt]*(1-float32(fraction)) + input[srcIndexInt+1]*float32(fraction)
		}
	}

	return output
}

// downsample performs simple decimation downsampling
func (w *PCM16Writer) downsample(input []float32, inputRate, outputRate int) []float32 {
	if inputRate == outputRate {
		return input
	}

	ratio := float64(inputRate) / float64(outputRate)
	outputLen := int(float64(len(input)) / ratio)
	output := make([]float32, outputLen)

	w.log.Debugw("downsampling audio",
		"inputRate", inputRate,
		"outputRate", outputRate,
		"ratio", ratio,
		"inputLen", len(input),
		"outputLen", outputLen)

	for i := range output {
		srcIndex := int(float64(i) * ratio)
		if srcIndex < len(input) {
			output[i] = input[srcIndex]
		}
	}

	return output
}
