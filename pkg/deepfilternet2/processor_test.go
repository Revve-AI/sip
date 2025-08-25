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
	"os"
	"testing"

	"github.com/livekit/protocol/logger"
	"github.com/stretchr/testify/require"
)

func TestProcessorCreation(t *testing.T) {
	log := logger.GetLogger()

	t.Run("processor creation without models", func(t *testing.T) {
		// This should fail gracefully since models won't exist
		processor, err := NewProcessor("./nonexistent-models", log)
		require.Error(t, err)
		require.Nil(t, processor)
	})

	t.Run("processor creation with mock directory", func(t *testing.T) {
		// Create a temporary directory for testing
		tmpDir := t.TempDir()

		// This should fail gracefully since ONNX models won't exist
		processor, err := NewProcessor(tmpDir, log)
		require.Error(t, err)
		require.Nil(t, processor)
	})
}

func TestProcessorWithMockModels(t *testing.T) {
	// Skip if we don't have access to real ONNX Runtime
	if os.Getenv("ONNX_RUNTIME_AVAILABLE") != "true" {
		t.Skip("ONNX Runtime not available in test environment")
	}

	log := logger.GetLogger()
	modelsPath := "./test-models"

	// This test would run if real models were available
	processor, err := NewProcessor(modelsPath, log)
	if err != nil {
		t.Skipf("Models not available: %v", err)
	}

	if processor != nil {
		defer processor.Close()

		// Test basic audio processing
		testSamples := make([]float32, HopSize)
		for i := range testSamples {
			testSamples[i] = float32(i) / float32(len(testSamples))
		}

		processed := processor.ProcessAudio(testSamples)
		require.Equal(t, len(testSamples), len(processed))
	}
}

func TestAudioProcessing(t *testing.T) {
	log := logger.GetLogger()

	// Test with a mock processor that bypasses ONNX
	processor := &Processor{
		audioBuffer: make([]float32, HopSize),
		log:         log,
		initialized: false, // Not initialized, so should bypass processing
	}

	testSamples := make([]float32, 480) // 10ms at 48kHz
	for i := range testSamples {
		testSamples[i] = 0.5 // Simple test signal
	}

	// Should return original samples when not initialized
	processed := processor.ProcessAudio(testSamples)
	require.Equal(t, testSamples, processed)
}
