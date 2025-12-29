"""Pytest configuration - runs before test collection."""
import sys

# Add required paths before any imports
sys.path.insert(0, '/opt/megatron-lm')  # megatron.core
sys.path.insert(0, '/opt/Megatron-Bridge/src')  # megatron.bridge
