package sip

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/livekit/protocol/logger"
	"github.com/livekit/sip/pkg/config"
)

func TestNewRoom(t *testing.T) {
	log := logger.GetLogger()
	stats := &RoomStats{}

	t.Run("NewRoom without RNNoise", func(t *testing.T) {
		conf := &config.Config{
			EnableRNNoise: false,
		}

		room := NewRoom(log, stats, conf)
		require.NotNil(t, room)
		require.NotNil(t, room.mix)
		require.NotNil(t, room.out)
	})

	t.Run("NewRoom with RNNoise enabled", func(t *testing.T) {
		conf := &config.Config{
			EnableRNNoise: true,
		}

		room := NewRoom(log, stats, conf)
		require.NotNil(t, room)
		require.NotNil(t, room.mix)
		require.NotNil(t, room.out)
		// Note: RNNoise integration test would require the rnnoise library to be installed
		// The test will gracefully fall back if RNNoise fails to initialize
		t.Logf("Room created successfully with RNNoise config enabled")
	})

	t.Run("NewRoom with nil config", func(t *testing.T) {
		room := NewRoom(log, stats, nil)
		require.NotNil(t, room)
		require.NotNil(t, room.mix)
		require.NotNil(t, room.out)
	})
}
