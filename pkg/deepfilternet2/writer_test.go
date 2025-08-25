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
	"testing"

	msdk "github.com/livekit/media-sdk"
	"github.com/livekit/protocol/logger"
	"github.com/stretchr/testify/require"
)

type mockWriter struct {
	samples    []msdk.PCM16Sample
	sampleRate int
}

func (m *mockWriter) WriteSample(sample msdk.PCM16Sample) error {
	m.samples = append(m.samples, sample)
	return nil
}

func (m *mockWriter) String() string {
	return "MockWriter"
}

func (m *mockWriter) SampleRate() int {
	return m.sampleRate
}

func TestPCM16WriterCreation(t *testing.T) {
	log := logger.GetLogger()

	t.Run("writer creation fails with invalid models path", func(t *testing.T) {
		mockNext := &mockWriter{sampleRate: 16000}

		writer, err := NewPCM16Writer(mockNext, "./nonexistent", log)
		require.Error(t, err)
		require.Nil(t, writer)
	})
}

func TestSampleConversion(t *testing.T) {
	log := logger.GetLogger()
	mockNext := &mockWriter{sampleRate: 16000}

	// Create a writer that will fail to initialize DeepFilterNet2 but still work for testing conversions
	writer := &PCM16Writer{
		next:       mockNext,
		log:        log,
		sampleRate: 16000,
		processor:  nil, // No processor for testing
	}

	t.Run("int16 to float32 conversion", func(t *testing.T) {
		testSamples := msdk.PCM16Sample{-32768, -16384, 0, 16384, 32767}
		expected := []float32{-1.0, -0.5, 0.0, 0.5, 0.9999694824}

		result := writer.int16ToFloat32(testSamples)
		require.Len(t, result, len(expected))

		for i, exp := range expected {
			require.InDelta(t, exp, result[i], 0.001, "Sample %d", i)
		}
	})

	t.Run("float32 to int16 conversion", func(t *testing.T) {
		testSamples := []float32{-1.0, -0.5, 0.0, 0.5, 1.0}
		expected := msdk.PCM16Sample{-32767, -16383, 0, 16383, 32767}

		result := writer.float32ToInt16(testSamples)
		require.Len(t, result, len(expected))

		for i, exp := range expected {
			require.InDelta(t, exp, result[i], 1, "Sample %d", i)
		}
	})

	t.Run("float32 to int16 with clamping", func(t *testing.T) {
		testSamples := []float32{-2.0, 2.0, -1.5, 1.5}
		expected := msdk.PCM16Sample{-32767, 32767, -32767, 32767}

		result := writer.float32ToInt16(testSamples)
		require.Equal(t, expected, result)
	})
}

func TestResampling(t *testing.T) {
	log := logger.GetLogger()
	mockNext := &mockWriter{sampleRate: 16000}

	writer := &PCM16Writer{
		next:       mockNext,
		log:        log,
		sampleRate: 16000,
		processor:  nil,
	}

	t.Run("upsampling 16kHz to 48kHz", func(t *testing.T) {
		input := []float32{0.0, 0.5, 1.0, 0.5, 0.0}
		result := writer.upsample(input, 16000, 48000)

		// Should be 3x longer
		require.Len(t, result, 15)

		// First and last samples should be preserved
		require.Equal(t, input[0], result[0])
		require.InDelta(t, input[len(input)-1], result[len(result)-1], 0.1)
	})

	t.Run("downsampling 48kHz to 16kHz", func(t *testing.T) {
		input := make([]float32, 15)
		for i := range input {
			input[i] = float32(i) / float32(len(input)-1)
		}

		result := writer.downsample(input, 48000, 16000)

		// Should be 1/3 the length
		require.Len(t, result, 5)

		// First sample should be preserved
		require.Equal(t, input[0], result[0])
	})

	t.Run("no resampling when rates match", func(t *testing.T) {
		input := []float32{0.1, 0.2, 0.3, 0.4, 0.5}

		upResult := writer.upsample(input, 16000, 16000)
		downResult := writer.downsample(input, 16000, 16000)

		require.Equal(t, input, upResult)
		require.Equal(t, input, downResult)
	})
}
